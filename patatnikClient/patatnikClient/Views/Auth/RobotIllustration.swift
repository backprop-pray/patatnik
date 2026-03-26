import SwiftUI

struct RobotIllustration: View {
    @State private var floating = false

    var body: some View {
        ZStack {
            // Subtle radial glow behind robot
            RadialGradient(
                colors: [Color(hex: 0xFF9500).opacity(0.05), .clear],
                center: .center,
                startRadius: 10,
                endRadius: 140
            )

            // Floating ambient dots
            FloatingDot(size: 6, xOffset: -70, yOffset: -50, delay: 0)
            FloatingDot(size: 4, xOffset: 75, yOffset: -30, delay: 0.8)
            FloatingDot(size: 8, xOffset: -55, yOffset: 40, delay: 1.6)
            FloatingDot(size: 5, xOffset: 65, yOffset: 55, delay: 2.4)

            // Robot canvas
            Canvas { context, size in
                let midX = size.width / 2
                let midY = size.height / 2

                // --- Tracks (two dark rounded rects at bottom) ---
                let trackWidth: CGFloat = 30
                let trackHeight: CGFloat = 50
                let trackY = midY + 30
                let trackSpacing: CGFloat = 80

                let leftTrack = RoundedRectangle(cornerRadius: 6).path(in: CGRect(
                    x: midX - trackSpacing / 2 - trackWidth,
                    y: trackY,
                    width: trackWidth,
                    height: trackHeight
                ))
                context.fill(leftTrack, with: .color(Color(hex: 0x1C1C1E)))
                context.stroke(leftTrack, with: .color(.white.opacity(0.08)), lineWidth: 1)

                let rightTrack = RoundedRectangle(cornerRadius: 6).path(in: CGRect(
                    x: midX + trackSpacing / 2,
                    y: trackY,
                    width: trackWidth,
                    height: trackHeight
                ))
                context.fill(rightTrack, with: .color(Color(hex: 0x1C1C1E)))
                context.stroke(rightTrack, with: .color(.white.opacity(0.08)), lineWidth: 1)

                // --- Chassis body (wider silver rect on tracks) ---
                let chassisWidth: CGFloat = 110
                let chassisHeight: CGFloat = 50
                let chassisY = trackY - chassisHeight + 10

                let chassis = RoundedRectangle(cornerRadius: 8).path(in: CGRect(
                    x: midX - chassisWidth / 2,
                    y: chassisY,
                    width: chassisWidth,
                    height: chassisHeight
                ))
                context.fill(chassis, with: .color(Color(white: 0.55)))
                context.stroke(chassis, with: .color(.white.opacity(0.15)), lineWidth: 1)

                // --- Raspberry Pi board (dark green rect on chassis) ---
                let piWidth: CGFloat = 40
                let piHeight: CGFloat = 26
                let piRect = RoundedRectangle(cornerRadius: 4).path(in: CGRect(
                    x: midX - piWidth / 2,
                    y: chassisY + (chassisHeight - piHeight) / 2,
                    width: piWidth,
                    height: piHeight
                ))
                context.fill(piRect, with: .color(Color(hex: 0x1B5E20)))
                context.stroke(piRect, with: .color(Color(hex: 0x2E7D32).opacity(0.6)), lineWidth: 0.5)

                // --- Ultrasonic sensor eyes (two glowing orange circles) ---
                let eyeRadius: CGFloat = 8
                let eyeSpacing: CGFloat = 30
                let eyeY = chassisY + chassisHeight / 2

                for xOff in [-eyeSpacing / 2, eyeSpacing / 2] {
                    let eyeCenter = CGPoint(x: midX + xOff, y: eyeY)
                    let eyeRect = CGRect(
                        x: eyeCenter.x - eyeRadius,
                        y: eyeCenter.y - eyeRadius,
                        width: eyeRadius * 2,
                        height: eyeRadius * 2
                    )
                    let eyePath = Circle().path(in: eyeRect)

                    // Glow
                    let glowRect = CGRect(
                        x: eyeCenter.x - eyeRadius * 2.5,
                        y: eyeCenter.y - eyeRadius * 2.5,
                        width: eyeRadius * 5,
                        height: eyeRadius * 5
                    )
                    let glowPath = Circle().path(in: glowRect)
                    context.fill(glowPath, with: .radialGradient(
                        Gradient(colors: [
                            Color(hex: 0xFF9500).opacity(0.4),
                            Color(hex: 0xFF9500).opacity(0.0)
                        ]),
                        center: eyeCenter,
                        startRadius: 0,
                        endRadius: eyeRadius * 2.5
                    ))

                    context.fill(eyePath, with: .radialGradient(
                        Gradient(colors: [
                            Color(hex: 0xFF9500),
                            Color(hex: 0xFF6B00)
                        ]),
                        center: eyeCenter,
                        startRadius: 0,
                        endRadius: eyeRadius
                    ))
                }

                // --- Webcam pole (thin vertical line from chassis top center) ---
                let poleBottom = chassisY
                let poleTop = chassisY - 45
                var polePath = Path()
                polePath.move(to: CGPoint(x: midX, y: poleBottom))
                polePath.addLine(to: CGPoint(x: midX, y: poleTop))
                context.stroke(polePath, with: .color(Color(white: 0.45)), lineWidth: 2.5)

                // --- Webcam head (circle at top of pole) ---
                let camRadius: CGFloat = 12
                let camCenter = CGPoint(x: midX, y: poleTop)
                let camRect = CGRect(
                    x: camCenter.x - camRadius,
                    y: camCenter.y - camRadius,
                    width: camRadius * 2,
                    height: camRadius * 2
                )
                let camPath = Circle().path(in: camRect)
                context.fill(camPath, with: .color(Color(white: 0.5)))
                context.stroke(camPath, with: .color(.white.opacity(0.15)), lineWidth: 1)

                // --- Lens (small dark circle inside webcam) ---
                let lensRadius: CGFloat = 5
                let lensRect = CGRect(
                    x: camCenter.x - lensRadius,
                    y: camCenter.y - lensRadius,
                    width: lensRadius * 2,
                    height: lensRadius * 2
                )
                let lensPath = Circle().path(in: lensRect)
                context.fill(lensPath, with: .color(Color(hex: 0x0A0A0F)))
            }
            .frame(width: 160, height: 180)
        }
        .frame(height: 200)
    }
}

// MARK: - Floating Dot

private struct FloatingDot: View {
    let size: CGFloat
    let xOffset: CGFloat
    let yOffset: CGFloat
    let delay: Double

    @State private var animating = false

    var body: some View {
        Circle()
            .fill(Color(hex: 0xFF9500).opacity(0.2))
            .frame(width: size, height: size)
            .offset(x: xOffset, y: yOffset + (animating ? -8 : 0))
            .animation(
                .easeInOut(duration: 3.5)
                .repeatForever(autoreverses: true)
                .delay(delay),
                value: animating
            )
            .onAppear { animating = true }
    }
}

// MARK: - Hex Color Extension

extension Color {
    init(hex: UInt, opacity: Double = 1.0) {
        self.init(
            red: Double((hex >> 16) & 0xFF) / 255,
            green: Double((hex >> 8) & 0xFF) / 255,
            blue: Double(hex & 0xFF) / 255,
            opacity: opacity
        )
    }
}

#Preview {
    ZStack {
        Color(hex: 0x0A0A0F).ignoresSafeArea()
        VStack(spacing: 8) {
            RobotIllustration()
            Text("TankPlant")
                .font(.title.bold())
                .foregroundStyle(.white)
            Text("Autonomous Plant Intelligence")
                .font(.subheadline).italic()
                .foregroundStyle(.secondary)
        }
    }
}
