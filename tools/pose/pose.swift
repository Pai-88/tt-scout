// Body pose for every person in a video, on-device with Apple's Vision framework (VNDetectHumanBodyPoseRequest, 19 joints, 2D).
// Used for tt_scout's posture overlay. Nothing is downloaded and no frame leaves the machine.
//
//   swiftc -O tools/pose/pose.swift -o tools/pose/pose
//   tools/pose/pose data/own/IMG_3144.mp4 out_own/IMG_3144_pose.csv [--step 2] [--windows windows.txt]
//
// windows.txt: one "t0 t1" pair of seconds per line; only frames inside a window are analysed (the points), which saves most of the time.
// Output CSV: frame,t,person,<joint>_x,<joint>_y,<joint>_c for 19 joints; pixels from the top-left of the full frame, c = Vision's
// confidence (0 = not found).
import AVFoundation
import Foundation
import Vision

let joints: [(String, VNHumanBodyPoseObservation.JointName)] = [
    ("nose", .nose), ("leye", .leftEye), ("reye", .rightEye), ("lear", .leftEar), ("rear", .rightEar),
    ("neck", .neck), ("lsho", .leftShoulder), ("rsho", .rightShoulder), ("lelb", .leftElbow), ("relb", .rightElbow),
    ("lwri", .leftWrist), ("rwri", .rightWrist), ("root", .root), ("lhip", .leftHip), ("rhip", .rightHip),
    ("lkne", .leftKnee), ("rkne", .rightKnee), ("lank", .leftAnkle), ("rank", .rightAnkle),
]

var args = Array(CommandLine.arguments.dropFirst())
func flag(_ name: String) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    let v = args[i + 1]; args.removeSubrange(i...(i + 1)); return v
}
let step = Int(flag("--step") ?? "1") ?? 1
var windows: [(Double, Double)] = []
if let wpath = flag("--windows"), let text = try? String(contentsOfFile: wpath, encoding: .utf8) {
    for line in text.split(separator: "\n") {
        let p = line.split(separator: " ").compactMap { Double($0) }
        if p.count == 2 { windows.append((p[0], p[1])) }
    }
}
// regions of interest in Vision's normalised coordinates (origin bottom-left): "full", or "halves" (default) = left and right 56 %
let regionMode = flag("--regions") ?? "halves"
// or "x0,x1,y0,y1;..." in image fractions from the top-left (e.g. the two ends of the table, below the top of the players' heads)
func parseRegions(_ spec: String) -> [CGRect] {
    if spec == "full" { return [CGRect(x: 0, y: 0, width: 1, height: 1)] }
    if spec == "halves" { return [CGRect(x: 0, y: 0, width: 0.56, height: 1), CGRect(x: 0.44, y: 0, width: 0.56, height: 1)] }
    return spec.split(separator: ";").compactMap { part -> CGRect? in
        let v = part.split(separator: ",").compactMap { Double($0) }
        guard v.count == 4 else { return nil }
        return CGRect(x: v[0], y: 1 - v[3], width: v[1] - v[0], height: v[3] - v[2])     // to Vision's bottom-left origin
    }
}
let regions = parseRegions(regionMode)
guard args.count == 2 else { print("usage: pose <video> <out.csv> [--step N] [--windows file]"); exit(2) }
let url = URL(fileURLWithPath: args[0])
let asset = AVURLAsset(url: url)
let sem = DispatchSemaphore(value: 0)
var track: AVAssetTrack?
var fps: Float = 30
Task {
    track = try? await asset.loadTracks(withMediaType: .video).first
    if let t = track { fps = (try? await t.load(.nominalFrameRate)) ?? 30 }
    sem.signal()
}
sem.wait()
guard let vtrack = track else { print("no video track"); exit(1) }
let reader = try AVAssetReader(asset: asset)
let output = AVAssetReaderTrackOutput(track: vtrack, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
output.alwaysCopiesSampleData = false
reader.add(output)
reader.startReading()

FileManager.default.createFile(atPath: args[1], contents: nil)
let out = FileHandle(forWritingAtPath: args[1])!
var header = "frame,t,person"
for (n, _) in joints { header += ",\(n)_x,\(n)_y,\(n)_c" }
out.write((header + "\n").data(using: .utf8)!)

func inWindow(_ t: Double) -> Bool {
    if windows.isEmpty { return true }
    for (a, b) in windows where t >= a && t <= b { return true }
    return false
}
var frame = 0, analysed = 0
let t0 = Date()
while let sample = output.copyNextSampleBuffer() {
    defer { frame += 1 }
    let t = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sample))
    if frame % step != 0 || !inWindow(t) { continue }
    guard let pb = CMSampleBufferGetImageBuffer(sample) else { continue }
    let W = Double(CVPixelBufferGetWidth(pb)), H = Double(CVPixelBufferGetHeight(pb))
    // The model works at a small input size, so the whole 16:9 frame makes each player tiny. Each region below is analysed on its own
    // (players are at the two ends of the table: the left and right halves, overlapping in the middle), person ids run on across them.
    var lines = ""
    var pi = 0
    for roi in regions {
        let req = VNDetectHumanBodyPoseRequest()
        req.regionOfInterest = roi
        let handler = VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up, options: [:])
        do { try handler.perform([req]) } catch { continue }
        for obs in req.results ?? [] {
            guard let pts = try? obs.recognizedPoints(.all) else { continue }
            var row = "\(frame),\(String(format: "%.4f", t)),\(pi)"
            for (_, j) in joints {
                if let p = pts[j], p.confidence > 0 {
                    // points come normalised to the region of interest, origin bottom-left
                    let x = (roi.origin.x + p.location.x * roi.width) * W
                    let y = (1 - (roi.origin.y + p.location.y * roi.height)) * H
                    row += String(format: ",%.1f,%.1f,%.3f", x, y, p.confidence)
                } else { row += ",,,0" }
            }
            lines += row + "\n"; pi += 1
        }
    }
    if !lines.isEmpty { out.write(lines.data(using: .utf8)!) }
    analysed += 1
    if analysed % 500 == 0 {
        let el = Date().timeIntervalSince(t0)
        FileHandle.standardError.write("\(analysed) frames analysed (video t=\(String(format: "%.1f", t)) s), \(String(format: "%.1f", Double(analysed) / el)) frames/s\n".data(using: .utf8)!)
    }
}
out.closeFile()
print("\(analysed) frames analysed of \(frame), fps \(fps) -> \(args[1])")
