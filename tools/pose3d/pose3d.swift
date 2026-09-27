// 3D body pose at chosen frames of a video, on-device with Apple's Vision framework (VNDetectHumanBodyPose3DRequest, 17 joints in
// metres), for tt_scout's after-match 3D view of the bodies and the table.
// Nothing is downloaded and no frame leaves the machine.
//
//   swiftc -O tools/pose3d/pose3d.swift -o tools/pose3d/pose3d
//   tools/pose3d/pose3d <video> <requests.txt> <out.jsonl> --f 974.5
//
// requests.txt: one "frame x0 y0 x1 y1 id" per line (pixels from the top-left of the full frame): analyse that frame, cropped to that box
// (the hitter), with the camera's focal length f and principal point at the picture's centre, shifted into the crop.
// Output, one JSON object per request answered: id, frame, t, height (Vision's estimate; "measured" false = its 1.8 m reference),
// joints {name: [x, y, z]} relative to the root joint (metres, y up), img {name: [u, v]} where Vision puts each joint in the full
// frame (pixels, top-left origin).
import AVFoundation
import CoreImage
import Foundation
import Vision
import VideoToolbox
import simd

var args = Array(CommandLine.arguments.dropFirst())
func flag(_ name: String) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    let v = args[i + 1]; args.removeSubrange(i...(i + 1)); return v
}
let focal = Double(flag("--f") ?? "0") ?? 0
guard args.count == 3 else { print("usage: pose3d <video> <requests.txt> <out.jsonl> --f <focal px>"); exit(2) }
struct Req { let frame: Int; let box: CGRect; let id: String }
var reqs: [Int: [Req]] = [:]
for line in (try String(contentsOfFile: args[1], encoding: .utf8)).split(separator: "\n") {
    let p = line.split(separator: " ")
    guard p.count == 6, let f = Int(p[0]), let x0 = Double(p[1]), let y0 = Double(p[2]), let x1 = Double(p[3]), let y1 = Double(p[4]) else { continue }
    reqs[f, default: []].append(Req(frame: f, box: CGRect(x: x0, y: y0, width: x1 - x0, height: y1 - y0), id: String(p[5])))
}
let last = reqs.keys.max() ?? -1
let asset = AVURLAsset(url: URL(fileURLWithPath: args[0]))
let sem = DispatchSemaphore(value: 0)
var track: AVAssetTrack?
Task { track = try? await asset.loadTracks(withMediaType: .video).first; sem.signal() }
sem.wait()
guard let vtrack = track else { print("no video track"); exit(1) }
let reader = try AVAssetReader(asset: asset)
let output = AVAssetReaderTrackOutput(track: vtrack, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
output.alwaysCopiesSampleData = false
reader.add(output)
reader.startReading()
FileManager.default.createFile(atPath: args[2], contents: nil)
let out = FileHandle(forWritingAtPath: args[2])!
let names: [(String, VNHumanBodyPose3DObservation.JointName)] = [
    ("top_head", .topHead), ("center_head", .centerHead), ("center_shoulder", .centerShoulder), ("left_shoulder", .leftShoulder),
    ("right_shoulder", .rightShoulder), ("left_elbow", .leftElbow), ("right_elbow", .rightElbow), ("left_wrist", .leftWrist),
    ("right_wrist", .rightWrist), ("spine", .spine), ("root", .root), ("left_hip", .leftHip), ("right_hip", .rightHip),
    ("left_knee", .leftKnee), ("right_knee", .rightKnee), ("left_ankle", .leftAnkle), ("right_ankle", .rightAnkle),
]
var frame = 0, done = 0, found = 0
let t0 = Date()
while let sample = output.copyNextSampleBuffer() {
    defer { frame += 1 }
    if frame > last { break }
    guard let rs = reqs[frame], let pb = CMSampleBufferGetImageBuffer(sample) else { continue }
    let t = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sample))
    var cgFull: CGImage?
    VTCreateCGImageFromCVPixelBuffer(pb, options: nil, imageOut: &cgFull)
    guard let cg = cgFull else { continue }
    let W = Double(cg.width), H = Double(cg.height)
    let f = focal > 0 ? focal : 0.9 * W
    for r in rs {
        let box = r.box.intersection(CGRect(x: 0, y: 0, width: W, height: H)).integral
        guard box.width > 32, box.height > 32, let crop = cg.cropping(to: box) else { continue }
        var K = matrix_float3x3(columns: (SIMD3<Float>(Float(f), 0, 0), SIMD3<Float>(0, Float(f), 0),
                                          SIMD3<Float>(Float(W / 2 - box.minX), Float(H / 2 - box.minY), 1)))
        let kdata = Data(bytes: &K, count: MemoryLayout<matrix_float3x3>.size)
        let handler = VNImageRequestHandler(cgImage: crop, options: [VNImageOption.cameraIntrinsics: kdata as CFData])
        let req = VNDetectHumanBodyPose3DRequest()
        do { try handler.perform([req]) } catch { continue }
        done += 1
        guard let o = req.results?.first else { continue }
        var js: [String] = [], im: [String] = []
        for (n, j) in names {
            guard let p = try? o.recognizedPoint(j), let q = try? o.pointInImage(j) else { continue }
            let c = p.position.columns.3
            js.append("\"\(n)\":[\(String(format: "%.4f,%.4f,%.4f", c.x, c.y, c.z))]")
            let u = box.minX + q.x * box.width, v = box.minY + (1 - q.y) * box.height
            im.append("\"\(n)\":[\(String(format: "%.1f,%.1f", u, v))]")
        }
        let line = "{\"id\":\"\(r.id)\",\"frame\":\(frame),\"t\":\(String(format: "%.4f", t)),\"height\":\(String(format: "%.3f", o.bodyHeight)),"
            + "\"measured\":\(o.heightEstimation == .measured),\"n\":\(req.results?.count ?? 0),\"joints\":{\(js.joined(separator: ","))},\"img\":{\(im.joined(separator: ","))}}\n"
        out.write(line.data(using: .utf8)!)
        found += 1
    }
    if done > 0 && done % 200 == 0 {
        FileHandle.standardError.write("\(done) crops analysed, \(String(format: "%.1f", Double(done) / Date().timeIntervalSince(t0))) per s\n".data(using: .utf8)!)
    }
}
out.closeFile()
print("\(found) bodies from \(done) crops (\(reqs.values.map { $0.count }.reduce(0, +)) requested) in \(String(format: "%.0f", Date().timeIntervalSince(t0))) s -> \(args[2])")
