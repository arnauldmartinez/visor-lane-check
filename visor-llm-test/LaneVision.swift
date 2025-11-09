// LaneVision.swift
// Camera + YOLOPv2 lane segmentation + parabola overlay helpers

import Foundation
import UIKit
import AVFoundation
import CoreVideo
import CoreImage
import CoreML
import Accelerate
import Combine   // for @Published / ObservableObject
import SwiftUI

// ===================== Model loader =====================
final class ModelManager {
    static let shared = ModelManager()
    private(set) var model: MLModel?
    private init() {}
    func loadIfNeeded(named baseName: String = "yolopv2") throws {
        if model != nil { return }
        let bundle = Bundle.main
        let cfg = MLModelConfiguration()
        if let compiled = bundle.url(forResource: baseName, withExtension: "mlmodelc") {
            model = try MLModel(contentsOf: compiled, configuration: cfg)
        } else if let packaged = bundle.url(forResource: baseName, withExtension: "mlpackage") {
            model = try MLModel(contentsOf: packaged, configuration: cfg)
        } else if let raw = bundle.url(forResource: baseName, withExtension: "mlmodel") {
            let compiledURL = try MLModel.compileModel(at: raw)
            model = try MLModel(contentsOf: compiledURL, configuration: cfg)
        } else {
            throw NSError(domain: "ModelLoad", code: -404, userInfo: [NSLocalizedDescriptionKey: "Model not found"])
        }
    }
}

enum InferenceUtils {
    static func requiredInputSize(from model: MLModel) -> CGSize? {
        guard let fd = model.modelDescription.inputDescriptionsByName.values.first else { return nil }
        if fd.type == .image, let c = fd.imageConstraint, c.pixelsWide > 0, c.pixelsHigh > 0 {
            return CGSize(width: c.pixelsWide, height: c.pixelsHigh)
        }
        return nil
    }
    static func pickSegHeads(from output: MLFeatureProvider) throws -> (MLMultiArray, MLMultiArray) {
        var outs: [(String, MLMultiArray)] = []
        for name in output.featureNames {
            if let arr = output.featureValue(for: name)?.multiArrayValue,
               arr.shape.count == 4, arr.shape[0].intValue == 1 {
                outs.append((name, arr))
            }
        }
        guard outs.count >= 2 else { throw NSError(domain: "segheads", code: -1) }
        if let da = outs.first(where: { $0.1.shape[1].intValue == 2 })?.1,
           let ll = outs.first(where: { $0.1.shape[1].intValue == 1 })?.1 { return (da, ll) }
        let sorted = outs.sorted { $0.1.shape[1].intValue < $1.1.shape[1].intValue }
        return (sorted[1].1, sorted[0].1)
    }
}

// ===================== Simple 2D mask =====================
struct Mask2D { let h: Int; let w: Int; let data: [UInt8] }

// ===================== Letterbox & overlay & parabola =====================
enum UtilsBridge {
    static func letterbox(image: UIImage,
                          targetSize: CGSize,
                          padColor: UIColor = UIColor(red: 114/255, green: 114/255, blue: 114/255, alpha: 1),
                          stride: Int = 32,
                          auto: Bool = false,
                          scaleFill: Bool = false,
                          scaleUp: Bool = true) throws -> (image: UIImage, padTop: Int, padBottom: Int) {
        guard let cg = image.cgImage else { throw NSError(domain: "lb", code: -1) }
        let W0 = cg.width, H0 = cg.height
        let Wt = Int(targetSize.width), Ht = Int(targetSize.height)

        var r = min(Double(Ht) / Double(H0), Double(Wt) / Double(W0))
        if !scaleUp { r = min(r, 1.0) }
        let w1 = Int(round(Double(W0) * r))
        let h1 = Int(round(Double(H0) * r))
        let dw = Wt - w1
        let dh = Ht - h1
        let left = Int(round(Double(dw) / 2.0))
        let top = Int(round(Double(dh) / 2.0))
        let bottom = Ht - h1 - top

        let fmt = UIGraphicsImageRendererFormat()
        fmt.scale = image.scale
        fmt.opaque = true
        let out = UIGraphicsImageRenderer(size: CGSize(width: Wt, height: Ht), format: fmt).image { ctx in
            padColor.setFill()
            ctx.fill(CGRect(x: 0, y: 0, width: Wt, height: Ht))
            UIImage(cgImage: cg).draw(in: CGRect(x: left, y: top, width: w1, height: h1))
        }
        return (out, top, bottom)
    }

    struct ParabolaOut { let totalLanes: Int; let egoLane: Int; let path: [CGPoint] }

    /// Upward-opening parabola scan (screen y grows downward)
    static func parabolaLaneCount(daMask: Mask2D, llMask: Mask2D, egoX: Int, buildPath: Bool = true) -> ParabolaOut {
        let W = daMask.w, H = daMask.h
        let x0 = (egoX >= 0) ? egoX : (W / 2)
        let y0 = Int(Double(H) * 0.82)
        let xSpan = max(24, Int(Double(W) * 0.35))
        let yTop = Int(Double(H) * 0.22)
        let curvature: Double = 0.45
        let kBase = (y0 > yTop) ? (Double(y0 - yTop) / Double(xSpan * xSpan)) : 0.0
        let a = -curvature * kBase
        let stepX = 2
        let radius = 2
        @inline(__always) func clamp(_ v: Int, _ lo: Int, _ hi: Int) -> Int { max(lo, min(hi, v)) }
        @inline(__always) func isBarrier(_ xx: Int, _ yy: Int) -> Bool {
            var sum = 0
            let cx = clamp(xx, 0, W-1), cy = clamp(yy, 0, H-1)
            let xMin = max(0, cx - radius), xMax = min(W-1, cx + radius)
            let yMin = max(0, cy - radius), yMax = min(H-1, cy + radius)
            for y in yMin...yMax {
                let row = y * W
                var x = xMin
                while x <= xMax { sum &+= Int(llMask.data[row + x]); x &+= 1 }
            }
            return sum > 0
        }
        @inline(__always) func isDrivable(_ xx: Int, _ yy: Int) -> Bool {
            let cx = clamp(xx, 0, W-1), cy = clamp(yy, 0, H-1)
            return daMask.data[cy * W + cx] != 0
        }

        var path: [CGPoint] = buildPath ? [] : []
        var regions: [(startX: Int, endX: Int)] = []
        var inDrivable = false
        var curStart = 0
        var crossings = 0
        var lastWasBarrier = true

        var x = 0
        while x < W {
            let dx = Double(x - x0)
            let y = Int(a * dx * dx + Double(y0))
            let yc = clamp(y, 0, H-1)
            if buildPath { path.append(CGPoint(x: CGFloat(x), y: CGFloat(yc))) }

            let barrier = isBarrier(x, yc)
            let drive = isDrivable(x, yc) && !barrier

            if barrier {
                if inDrivable { regions.append((curStart, x - 1)); inDrivable = false }
            } else if drive {
                if !inDrivable {
                    inDrivable = true
                    curStart = x
                    if lastWasBarrier { crossings &+= 1 }
                }
            }
            lastWasBarrier = barrier
            x &+= stepX
        }
        if inDrivable { regions.append((curStart, W - 1)) }

        let totalLanes = max( (crossings > 0 ? crossings : (regions.isEmpty ? 0 : 1)), 0)
        var egoLane = 1
        if !regions.isEmpty {
            var bestIdx = 0
            var bestDist = Int.max
            for (i, r) in regions.enumerated() {
                if r.startX <= x0 && x0 <= r.endX { bestIdx = i; break }
                let d = (x0 < r.startX) ? (r.startX - x0) : (x0 - r.endX)
                if d < bestDist { bestDist = d; bestIdx = i }
            }
            egoLane = bestIdx + 1
        }

        return ParabolaOut(totalLanes: totalLanes, egoLane: egoLane, path: buildPath ? path : [])
    }

    static func showSegResult(baseBGR base: UIImage,
                              drivableMask: Mask2D,
                              laneMask: Mask2D) throws -> UIImage {
        guard let cg = base.cgImage else { throw NSError(domain: "overlay", code: -1) }
        let W = cg.width, H = cg.height

        // Extract base pixels
        var rgba = [UInt8](repeating: 0, count: W * H * 4)
        let cs = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(data: &rgba, width: W, height: H, bitsPerComponent: 8,
                            bytesPerRow: W * 4, space: cs,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)!
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: W, height: H))

        // Blend masks
        for i in 0..<(W*H) {
            let gOn = drivableMask.data[i] != 0
            let rOn = laneMask.data[i] != 0
            if gOn || rOn {
                let idx = i * 4
                var r = Float(rgba[idx + 0]), g = Float(rgba[idx + 1]), b = Float(rgba[idx + 2])
                if gOn { r = 0.5*r + 0.0;   g = 0.5*g + 127.5; b = 0.5*b + 0.0 }
                if rOn { r = 0.5*r + 127.5; g = 0.5*g + 0.0;   b = 0.5*b + 0.0 }
                rgba[idx + 0] = UInt8(max(0, min(255, Int(r))))
                rgba[idx + 1] = UInt8(max(0, min(255, Int(g))))
                rgba[idx + 2] = UInt8(max(0, min(255, Int(b))))
            }
        }

        let outCtx = CGContext(data: &rgba, width: W, height: H, bitsPerComponent: 8,
                               bytesPerRow: W * 4, space: cs,
                               bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)!
        let outCG = outCtx.makeImage()!
        return UIImage(cgImage: outCG, scale: base.scale, orientation: base.imageOrientation)
    }

    static func overlayParabola(on image: UIImage,
                                parabola: [CGPoint],
                                egoLane: Int,
                                totalLanes: Int) throws -> UIImage {
        let format = UIGraphicsImageRendererFormat()
        format.scale = image.scale; format.opaque = false
        return UIGraphicsImageRenderer(size: image.size, format: format).image { ctx in
            image.draw(in: CGRect(origin: .zero, size: image.size))

            if !parabola.isEmpty {
                let path = UIBezierPath()
                path.move(to: parabola[0])
                for p in parabola.dropFirst() { path.addLine(to: p) }
                path.lineWidth = 3
                UIColor.yellow.setStroke()
                path.stroke()
            }

            let baseAttrs: [NSAttributedString.Key: Any] = [
                .font: UIFont.monospacedSystemFont(ofSize: 16, weight: .medium),
                .foregroundColor: UIColor.white
            ]
            let strokeAttrs: [NSAttributedString.Key: Any] = [
                .font: UIFont.monospacedSystemFont(ofSize: 16, weight: .medium),
                .strokeColor: UIColor.black,
                .strokeWidth: -3.0
            ]
            let l1 = "Ego in lane \(max(1, egoLane))"
            let l2 = "\(max(0, totalLanes)) total lanes"
            let p1 = CGPoint(x: 8, y: 8)
            let p2 = CGPoint(x: 8, y: 30)
            (l1 as NSString).draw(at: p1, withAttributes: strokeAttrs)
            (l2 as NSString).draw(at: p2, withAttributes: strokeAttrs)
            (l1 as NSString).draw(at: p1, withAttributes: baseAttrs)
            (l2 as NSString).draw(at: p2, withAttributes: baseAttrs)
        }
    }
}

// ===================== FAST MLMultiArray → masks =====================
enum UtilsFast {
    static func makeMasksFast(drivable: MLMultiArray,
                              lane: MLMultiArray,
                              padTop: Int,
                              padBottom: Int) throws -> (Mask2D, Mask2D) {
        guard drivable.shape.count == 4, drivable.shape[0].intValue == 1,
              lane.shape.count == 4,      lane.shape[0].intValue == 1 else { throw NSError(domain: "mask", code: -1) }
        let Cda = drivable.shape[1].intValue
        let H    = drivable.shape[2].intValue
        let W    = drivable.shape[3].intValue
        let Cll  = lane.shape[1].intValue

        guard H > (padTop + padBottom), W >= 1 else { throw NSError(domain: "mask", code: -2) }
        let y0 = max(0, min(padTop, H))
        let y1 = max(y0, min(H - padBottom, H))
        let Hc = y1 - y0

        let daModel = try maskModelSize(from: drivable, C: Cda, H: H, W: W)
        let llModel = try maskModelSize(from: lane,      C: Cll, H: H, W: W)

        let daCrop = cropMask8(daModel, H: H, W: W, y0: y0, y1: y1)
        let llCrop = cropMask8(llModel, H: H, W: W, y0: y0, y1: y1)

        let Ho = Hc * 2, Wo = W * 2
        let daOut = try resizeMask8Nearest(daCrop, srcH: Hc, srcW: W, dstH: Ho, dstW: Wo)
        let llOut = try resizeMask8Nearest(llCrop, srcH: Hc, srcW: W, dstH: Ho, dstW: Wo)
        return (Mask2D(h: Ho, w: Wo, data: daOut), Mask2D(h: Ho, w: Wo, data: llOut))
    }

    private static func maskModelSize(from arr: MLMultiArray, C: Int, H: Int, W: Int) throws -> [UInt8] {
        var out = [UInt8](repeating: 0, count: H * W)
        let fp = UnsafeMutablePointer<Float32>(OpaquePointer(arr.dataPointer))
        if C == 1 {
            for i in 0..<(H*W) { out[i] = (fp[i].rounded() >= 1.0) ? 1 : 0 }
        } else if C == 2 {
            let base0 = 0, base1 = H * W
            for i in 0..<(H*W) { out[i] = (fp[base1 + i] > fp[base0 + i]) ? 1 : 0 }
        } else { throw NSError(domain: "mask", code: -3) }
        return out
    }

    private static func cropMask8(_ mask: [UInt8], H: Int, W: Int, y0: Int, y1: Int) -> [UInt8] {
        let Hc = y1 - y0
        var out = [UInt8](repeating: 0, count: Hc * W)
        for yy in 0..<Hc {
            let srcRow = (y0 + yy) * W
            let dstRow = yy * W
            out[dstRow..<(dstRow + W)] = mask[srcRow..<(srcRow + W)]
        }
        return out
    }

    private static func resizeMask8Nearest(_ src: [UInt8], srcH: Int, srcW: Int, dstH: Int, dstW: Int) throws -> [UInt8] {
        var dst = [UInt8](repeating: 0, count: dstH * dstW)
        try src.withUnsafeBytes { sRaw in
            guard let sBase = sRaw.baseAddress else { throw NSError(domain: "vImage", code: -1) }
            try dst.withUnsafeMutableBytes { dRaw in
                guard let dBase = dRaw.baseAddress else { throw NSError(domain: "vImage", code: -2) }
                var s = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: sBase),
                                      height: vImagePixelCount(srcH),
                                      width:  vImagePixelCount(srcW),
                                      rowBytes: srcW * MemoryLayout<UInt8>.size)
                var d = vImage_Buffer(data: dBase,
                                      height: vImagePixelCount(dstH),
                                      width:  vImagePixelCount(dstW),
                                      rowBytes: dstW * MemoryLayout<UInt8>.size)
                let err = vImageScale_Planar8(&s, &d, nil, vImage_Flags(kvImageNoFlags))
                if err != kvImageNoError { throw NSError(domain: "vImage", code: Int(err)) }
            }
        }
        return dst
    }
}

// ===================== Camera manager & preview =====================

final class CameraManager: NSObject, ObservableObject {
    let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "camera.queue")

    @Published var latestFrame: CGImage?
    var onFrame: ((CGImage) -> Void)?

    func start() {
        DispatchQueue.global(qos: .userInitiated).async {
            if self.session.inputs.isEmpty {
                self.session.beginConfiguration()
                self.session.sessionPreset = .high
                guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
                      let input = try? AVCaptureDeviceInput(device: device) else { return }
                if self.session.canAddInput(input) { self.session.addInput(input) }

                self.output.alwaysDiscardsLateVideoFrames = true
                self.output.setSampleBufferDelegate(self, queue: self.queue)
                self.output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
                if self.session.canAddOutput(self.output) { self.session.addOutput(self.output) }
                self.session.commitConfiguration()
            }
            self.session.startRunning()
        }
    }

    func stop() {
        DispatchQueue.global(qos: .userInitiated).async {
            if self.session.isRunning { self.session.stopRunning() }
        }
    }
}

// Move the delegate conformance to an extension and mark callback as `nonisolated`
extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {

    nonisolated func captureOutput(_ output: AVCaptureOutput,
                                   didOutput sampleBuffer: CMSampleBuffer,
                                   from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let ci = CIImage(cvPixelBuffer: pb)
        let ctx = CIContext(options: [.cacheIntermediates: false])
        if let cg = ctx.createCGImage(ci, from: ci.extent) {
            // Notify any off-main processing first
            self.onFrame?(cg)
            // Publish to SwiftUI on main
            DispatchQueue.main.async { [weak self] in
                self?.latestFrame = cg
            }
        }
    }
}

// ===================== SwiftUI Preview View =====================

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> UIView {
        let v = UIView()
        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspectFill
        v.layer.addSublayer(layer)
        context.coordinator.layer = layer
        return v
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        context.coordinator.layer?.frame = uiView.bounds
    }

    func makeCoordinator() -> Coord { Coord() }

    final class Coord {
        var layer: AVCaptureVideoPreviewLayer?
    }
}
