import Foundation
import CoreMotion
import Combine

public struct MotionSample: Codable, Sendable {
    public let timestamp: TimeInterval
    public let rotationRateX: Double
    public let rotationRateY: Double
    public let rotationRateZ: Double
    public let userAccelerationX: Double
    public let userAccelerationY: Double
    public let userAccelerationZ: Double
    public let gravityX: Double
    public let gravityY: Double
    public let gravityZ: Double
    public let yaw: Double
    public let roll: Double
    public let pitch: Double

    public init(
        timestamp: TimeInterval,
        rotationRateX: Double,
        rotationRateY: Double,
        rotationRateZ: Double,
        userAccelerationX: Double,
        userAccelerationY: Double,
        userAccelerationZ: Double,
        gravityX: Double,
        gravityY: Double,
        gravityZ: Double,
        yaw: Double,
        roll: Double,
        pitch: Double
    ) {
        self.timestamp = timestamp
        self.rotationRateX = rotationRateX
        self.rotationRateY = rotationRateY
        self.rotationRateZ = rotationRateZ
        self.userAccelerationX = userAccelerationX
        self.userAccelerationY = userAccelerationY
        self.userAccelerationZ = userAccelerationZ
        self.gravityX = gravityX
        self.gravityY = gravityY
        self.gravityZ = gravityZ
        self.yaw = yaw
        self.roll = roll
        self.pitch = pitch
    }
}

public struct MotionWindow: Sendable {
    public let samples: [MotionSample]

    public init(samples: [MotionSample]) {
        self.samples = samples
    }

    public var count: Int { samples.count }

    /// Core ML input channel order used by Code 1:
    /// accX, accY, accZ, gyroX, gyroY, gyroZ.
    public func modelInputMatrix() -> [[Double]] {
        samples.map { sample in
            [
                sample.userAccelerationX,
                sample.userAccelerationY,
                sample.userAccelerationZ,
                sample.rotationRateX,
                sample.rotationRateY,
                sample.rotationRateZ
            ]
        }
    }

    public func accelerationMatrix() -> [[Double]] {
        samples.map { [$0.userAccelerationX, $0.userAccelerationY, $0.userAccelerationZ] }
    }

    public func rotationRateMatrix() -> [[Double]] {
        samples.map { [$0.rotationRateX, $0.rotationRateY, $0.rotationRateZ] }
    }
}

public final class MotionRecorder: ObservableObject {
    public enum RecorderError: Error {
        case deviceMotionUnavailable
    }

    private let motionManager: CMMotionManager
    private let queue: OperationQueue
    private let sampleRateHz: Double
    private var buffer: [MotionSample]

    @Published public private(set) var isRecording: Bool

    public init(sampleRateHz: Double = 50.0) {
        self.motionManager = CMMotionManager()
        self.queue = OperationQueue()
        self.queue.qualityOfService = .userInitiated
        self.sampleRateHz = sampleRateHz
        self.buffer = []
        self.isRecording = false
    }

    public func start() throws {
        guard motionManager.isDeviceMotionAvailable else {
            throw RecorderError.deviceMotionUnavailable
        }

        buffer.removeAll(keepingCapacity: true)
        isRecording = true
        motionManager.deviceMotionUpdateInterval = 1.0 / sampleRateHz

        motionManager.startDeviceMotionUpdates(to: queue) { [weak self] motion, error in
            guard let self, let motion, error == nil else { return }

            let sample = MotionSample(
                timestamp: motion.timestamp,
                rotationRateX: motion.rotationRate.x,
                rotationRateY: motion.rotationRate.y,
                rotationRateZ: motion.rotationRate.z,
                userAccelerationX: motion.userAcceleration.x,
                userAccelerationY: motion.userAcceleration.y,
                userAccelerationZ: motion.userAcceleration.z,
                gravityX: motion.gravity.x,
                gravityY: motion.gravity.y,
                gravityZ: motion.gravity.z,
                yaw: motion.attitude.yaw,
                roll: motion.attitude.roll,
                pitch: motion.attitude.pitch
            )

            self.buffer.append(sample)
        }
    }

    @discardableResult
    public func stop() -> [MotionSample] {
        motionManager.stopDeviceMotionUpdates()
        isRecording = false
        return buffer
    }

    public func currentSamples() -> [MotionSample] {
        buffer
    }
}
