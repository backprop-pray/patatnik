import SwiftUI

// MARK: - Color Extensions

extension Color {
    static let appOrange    = Color(red: 1.0,  green: 0.58, blue: 0.0)
    static let appDark      = Color(red: 0.04, green: 0.04, blue: 0.06)
    static let appSurface   = Color(red: 0.11, green: 0.11, blue: 0.12)
    static let appSecondary = Color(red: 0.56, green: 0.56, blue: 0.58)
}

// MARK: - RoundedCorner Shape

struct RoundedCorner: Shape {
    var radius: CGFloat
    var corners: UIRectCorner

    func path(in rect: CGRect) -> Path {
        let path = UIBezierPath(
            roundedRect: rect,
            byRoundingCorners: corners,
            cornerRadii: CGSize(width: radius, height: radius)
        )
        return Path(path.cgPath)
    }
}

// MARK: - Plant Detail Sheet

struct PlantDetailSheet: View {
    let plant: Plant
    let onClose: () -> Void

    @StateObject private var viewModel = PlantDetailViewModel()
    @State private var appeared = false
    @State private var shimmerOffset: CGFloat = -300
    @State private var dragOffset: CGFloat = 0
    @Environment(\.horizontalSizeClass) var sizeClass

    // MARK: - URL Normalization

    private func httpsURL(from raw: String?) -> URL? {
        guard var raw = raw, !raw.isEmpty else { return nil }
        if raw.hasPrefix("http://") {
            raw = "https://" + raw.dropFirst("http://".count)
        }
        if raw.hasPrefix("//") {
            raw = "https:" + raw
        }
        return URL(string: raw)
    }

    var body: some View {
        GeometryReader { geo in
            VStack { Spacer(); sheetCard(geo: geo) }
                .ignoresSafeArea(edges: .bottom)
                .onAppear {
                    appeared = true
                    viewModel.loadRecommendation(for: plant)
                }
                .onDisappear { viewModel.cancel() }
        }
        .animation(.spring(response: 0.4, dampingFraction: 0.82), value: appeared)
    }

    // MARK: - Sheet Card

    private func sheetCard(geo: GeometryProxy) -> some View {
        let maxH = geo.size.height * 0.58
        return sheetContent
            .frame(maxWidth: sizeClass == .regular ? 420 : .infinity)
            .frame(maxHeight: maxH)
            .background(.ultraThinMaterial)
            .clipShape(RoundedCorner(radius: 24, corners: [.topLeft, .topRight]))
            .overlay(alignment: .top) {
                RoundedCorner(radius: 24, corners: [.topLeft, .topRight])
                    .stroke(Color.white.opacity(0.2), lineWidth: 0.5)
            }
            .shadow(color: .black.opacity(0.3), radius: 30, x: 0, y: -8)
            .offset(y: dragOffset)
            .gesture(
                DragGesture()
                    .onChanged { value in
                        if value.translation.height > 0 {
                            dragOffset = value.translation.height
                        }
                    }
                    .onEnded { value in
                        if value.translation.height > 80 {
                            withAnimation(.spring(response: 0.35)) { onClose() }
                        } else {
                            withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
                                dragOffset = 0
                            }
                        }
                    }
            )
            .transition(.move(edge: .bottom).combined(with: .opacity))
    }

    // MARK: - Sheet Content

    private var sheetContent: some View {
        ScrollView(.vertical, showsIndicators: false) {
            VStack(spacing: 0) {
                // Drag handle
                RoundedRectangle(cornerRadius: 3)
                    .fill(Color.white.opacity(0.3))
                    .frame(width: 40, height: 5)
                    .padding(.top, 12)
                    .padding(.bottom, 8)

                // ZONE 1 — Hero image
                heroImage

                // ZONE 2 — Identity row
                identityRow
                    .padding(.top, 14)
                    .padding(.bottom, 12)

                // Separator
                separator

                // ZONE 3 — Analysis section
                PlantRecommendationView(viewModel: viewModel)

                // ZONE 4 — Opinion section (only when user has rejected)
                if viewModel.recommendationLoaded && viewModel.hasResponded && !viewModel.responseAccepted {
                    separator
                    OpinionInputView(viewModel: viewModel, plantId: plant.id)
                }

                Spacer().frame(height: 40)
            }
        }
    }

    // MARK: - Separator

    private var separator: some View {
        Rectangle()
            .fill(.primary.opacity(0.08))
            .frame(height: 1)
            .padding(.horizontal, 20)
    }

    // MARK: - ZONE 1: Hero Image

    private var heroImage: some View {
        // Use a fixed-height container that clips all overflow
        RoundedRectangle(cornerRadius: 16)
            .fill(Color(.systemGray5))
            .frame(height: 190)
            .overlay(
                AsyncImage(url: httpsURL(from: plant.imageUrl)) { phase in
                    switch phase {
                    case .empty:
                        shimmerView()
                    case .success(let image):
                        GeometryReader { geo in
                            image
                                .resizable()
                                .scaledToFill()
                                .frame(width: geo.size.width, height: geo.size.height)
                                .clipped()
                        }
                        .transition(.opacity.animation(.easeIn(duration: 0.25)))
                    case .failure:
                        VStack(spacing: 8) {
                            Image(systemName: "leaf.fill")
                                .font(.system(size: 32))
                                .foregroundStyle(Color.appOrange.opacity(0.5))
                            Text("No image available")
                                .font(.system(size: 12, weight: .regular))
                                .foregroundStyle(.secondary)
                        }
                    @unknown default:
                        EmptyView()
                    }
                }
            )
            .clipShape(RoundedRectangle(cornerRadius: 16))
            // Gradient overlay — transparent top → semi-dark bottom
            .overlay(alignment: .bottom) {
                LinearGradient(
                    colors: [.clear, .clear, .black.opacity(0.35)],
                    startPoint: .top,
                    endPoint: .bottom
                )
                .frame(height: 80)
                .clipShape(
                    RoundedCorner(radius: 16, corners: [.bottomLeft, .bottomRight])
                )
            }
            // Border
            .overlay(
                RoundedRectangle(cornerRadius: 16)
                    .stroke(Color.white.opacity(0.1), lineWidth: 0.5)
            )
            // Plant ID pill — bottom-left
            .overlay(alignment: .bottomLeading) {
                Text("#\(plant.id)")
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 5)
                    .background(
                        Capsule()
                            .fill(Color.appOrange)
                            .shadow(color: Color.appOrange.opacity(0.4), radius: 6, y: 2)
                    )
                    .padding(10)
            }
            // Close button — top-right
            .overlay(alignment: .topTrailing) {
                Button(action: onClose) {
                    ZStack {
                        Circle()
                            .fill(.ultraThinMaterial)
                            .frame(width: 34, height: 34)
                            .shadow(color: .black.opacity(0.25), radius: 6, y: 2)
                        Image(systemName: "xmark")
                            .font(.system(size: 12, weight: .bold))
                            .foregroundStyle(.white.opacity(0.9))
                    }
                    .frame(width: 44, height: 44)
                }
                .padding(4)
            }
            .shadow(color: .black.opacity(0.12), radius: 10, y: 4)
            .padding(.horizontal, 16)
    }

    // MARK: - ZONE 2: Identity Row

    private var identityRow: some View {
        VStack(spacing: 5) {
            HStack(spacing: 8) {
                Text("Plant #\(plant.id)")
                    .font(.system(size: 22, weight: .bold, design: .rounded))
                    .foregroundStyle(.primary)

                statusDot
            }

            HStack(spacing: 5) {
                Image(systemName: "mappin.circle.fill")
                    .font(.system(size: 12))
                    .foregroundStyle(Color.appOrange)
                Text(formattedDate(plant.createdAt))
                    .font(.system(size: 13, weight: .regular))
                    .foregroundStyle(.primary.opacity(0.6))
                    .lineLimit(1)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 18)
    }

    // MARK: - Status Dot

    private var statusDot: some View {
        ZStack {
            if viewModel.recommendationLoaded {
                // Green dot — analysis complete
                Circle()
                    .fill(.green)
                    .frame(width: 8, height: 8)
                    .transition(.scale.combined(with: .opacity))
            } else if viewModel.recommendationError {
                // Red dot — error
                Circle()
                    .fill(.red)
                    .frame(width: 8, height: 8)
                    .transition(.scale.combined(with: .opacity))
            } else {
                // Pulsing orange dot — analyzing
                PulsingDot()
            }
        }
        .animation(.spring(response: 0.5, dampingFraction: 0.8), value: viewModel.recommendationLoaded)
        .animation(.spring(response: 0.5, dampingFraction: 0.8), value: viewModel.recommendationError)
    }

    // MARK: - Shimmer

    private func shimmerView() -> some View {
        GeometryReader { geo in
            LinearGradient(
                colors: [.clear, Color.white.opacity(0.25), .clear],
                startPoint: .leading,
                endPoint: .trailing
            )
            .frame(width: geo.size.width * 0.6)
            .offset(x: shimmerOffset)
            .onAppear {
                withAnimation(.linear(duration: 1.2).repeatForever(autoreverses: false)) {
                    shimmerOffset = 300
                }
            }
        }
        .clipped()
    }

    // MARK: - Date Formatter

    private func formattedDate(_ isoString: String?) -> String {
        guard let isoString, !isoString.isEmpty else {
            return "Unknown date"
        }

        // Normalize: truncate fractional seconds to 3 digits (SSS) so
        // DateFormatter can parse Java's nanosecond-precision strings
        // e.g. "2026-03-27T13:45:22.123456789" → "2026-03-27T13:45:22.123"
        let normalized: String
        if let dotIndex = isoString.firstIndex(of: ".") {
            let fractionalStart = isoString.index(after: dotIndex)
            let digits = isoString[fractionalStart...].prefix(3)
            normalized = String(isoString[...dotIndex]) + digits
        } else {
            normalized = isoString
        }

        let display = DateFormatter()
        display.dateFormat = "'Discovered' MMM d, yyyy"

        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")

        // Pattern 1: with fractional seconds (normalized to 3 digits)
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSS"
        if let date = formatter.date(from: normalized) {
            return display.string(from: date)
        }

        // Pattern 2: no fractional seconds
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
        if let date = formatter.date(from: normalized) {
            return display.string(from: date)
        }

        // Pattern 3: date only
        formatter.dateFormat = "yyyy-MM-dd"
        if let date = formatter.date(from: normalized) {
            return display.string(from: date)
        }

        // Fallback: show the raw string so something always appears
        return isoString
    }
}

// MARK: - Pulsing Dot

private struct PulsingDot: View {
    @State private var isPulsing = false

    var body: some View {
        ZStack {
            Circle()
                .fill(Color.appOrange.opacity(0.2))
                .frame(width: 16, height: 16)
                .scaleEffect(isPulsing ? 1.4 : 1.0)
                .opacity(isPulsing ? 0 : 0.5)

            Circle()
                .fill(Color.appOrange)
                .frame(width: 8, height: 8)
                .shadow(color: Color.appOrange.opacity(0.4), radius: 3)
        }
        .frame(width: 16, height: 16)
        .onAppear {
            withAnimation(.easeInOut(duration: 1.2).repeatForever(autoreverses: false)) {
                isPulsing = true
            }
        }
    }
}

// MARK: - Preview

#Preview {
    ZStack {
        LinearGradient(colors: [.blue.opacity(0.6), .green.opacity(0.6)], startPoint: .top, endPoint: .bottom)
            .ignoresSafeArea()
        PlantDetailSheet(
            plant: Plant(
                id: 42,
                latitude: 42.69,
                longitude: 23.32,
                imageUrl: nil,
                userId: 1,
                createdAt: "2026-03-12T10:00:00",
                disease: "Tomato — Late blight",
                recommendedAction: "Rotate crops every 4 years, keep leaves dry",
                status: false,
                processedPlantId: 123
            ),
            onClose: {}
        )
    }
}
