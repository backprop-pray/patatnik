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
    @Environment(\.horizontalSizeClass) var sizeClass

    // MARK: - URL Normalization
    private func httpsURL(from raw: String?) -> URL? {
        guard var raw = raw, !raw.isEmpty else { return nil }
        // If scheme is http, upgrade to https (Cloudinary requires https on iOS by default)
        if raw.hasPrefix("http://") {
            raw = "https://" + raw.dropFirst("http://".count)
        }
        // Some backends may return protocol-relative URLs like //res.cloudinary.com/...
        if raw.hasPrefix("//") {
            raw = "https:" + raw
        }
        return URL(string: raw)
    }

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .topTrailing) {
                // Bottom pinned sheet content
                VStack { Spacer(); sheetCard(geo: geo) }
                    .ignoresSafeArea(edges: .bottom)

                // Floating close button (above the sheet)
                Button(action: onClose) {
                    ZStack {
                        Circle()
                            .fill(.regularMaterial)
                            .frame(width: 32, height: 32)
                            .shadow(color: .black.opacity(0.2), radius: 4, y: 2)
                        Image(systemName: "xmark")
                            .font(.system(size: 12, weight: .bold))
                            .foregroundStyle(Color.appSecondary)
                    }
                }
                .offset(x: -16, y: -16)
            }
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
        let maxH = UIScreen.main.bounds.height * 0.62
        return sheetContent
            .frame(maxWidth: sizeClass == .regular ? 420 : .infinity)
            .frame(maxHeight: maxH)
            .background(.regularMaterial)
            .clipShape(RoundedCorner(radius: 24, corners: [.topLeft, .topRight]))
            .shadow(color: .black.opacity(0.15), radius: 20, x: 0, y: -4)
            .transition(.move(edge: .bottom).combined(with: .opacity))
    }

    // MARK: - Sheet Content

    private var sheetContent: some View {
        ScrollView(.vertical, showsIndicators: false) {
            VStack(spacing: 0) {
                // 1. Drag handle
                RoundedRectangle(cornerRadius: 2)
                    .fill(Color.gray.opacity(0.35))
                    .frame(width: 36, height: 4)
                    .padding(.top, 12)
                    .padding(.bottom, 8)

                // 2. Image holder card
                imageCard

                // 3. Plant header (no close button inside)
                plantHeader

                // 4. Thin divider
                Divider().padding(.horizontal, 20)

                // 5. PlantRecommendationView (robot or recommendation)
                PlantRecommendationView(viewModel: viewModel)

                // 6. Divider (only when recommendationLoaded)
                if viewModel.recommendationLoaded {
                    Divider().padding(.horizontal, 20)
                }

                // 7. OpinionInputView (only when recommendationLoaded)
                if viewModel.recommendationLoaded {
                    OpinionInputView(viewModel: viewModel, plantId: plant.id)
                }

                // 8. Bottom spacer
                Spacer().frame(height: 40)
            }
        }
    }

    // MARK: - Image Card Treatment

    private var imageCard: some View {
        RoundedRectangle(cornerRadius: 16)
            .fill(Color.appSurface.opacity(0.3))
            .frame(height: 200)
            .overlay(
                AsyncImage(url: httpsURL(from: plant.imageUrl)) { phase in
                    switch phase {
                    case .empty:
                        ZStack {
                            Color.appSurface.opacity(0.5)
                            shimmerView()
                        }
                    case .success(let image):
                        image
                            .resizable()
                            .scaledToFill()
                            .transition(.opacity.animation(.easeIn(duration:0.3)))
                    case .failure:
                        ZStack {
                            Color.appSurface.opacity(0.5)
                            VStack(spacing: 8) {
                                Image(systemName: "leaf.fill")
                                    .font(.system(size: 32))
                                    .foregroundStyle(Color.appOrange)
                                Text("No image")
                                    .font(.caption)
                                    .foregroundStyle(Color.appSecondary)
                            }
                        }
                    @unknown default:
                        EmptyView()
                    }
                }
                .clipShape(RoundedRectangle(cornerRadius: 16))
            )
            .overlay(
                // Subtle border
                RoundedRectangle(cornerRadius: 16)
                    .stroke(Color.white.opacity(0.1), lineWidth: 1)
            )
            .shadow(color: .black.opacity(0.12), radius: 8, y: 4)
            .padding(.horizontal, 16)
            .padding(.top, 8)
            .padding(.bottom, 12)
    }

    // MARK: - Plant Header Cleanup

    private var plantHeader: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Plant #\(plant.id)")
                .font(.title2.bold())
                .foregroundStyle(Color.appDark)

            HStack(spacing: 4) {
                Image(systemName: "clock")
                    .font(.caption)
                    .foregroundStyle(Color.appSecondary)
                Text(formattedDate(plant.createdAt))
                    .font(.caption)
                    .foregroundStyle(Color.appSecondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 20)
        .padding(.top, 12)
        .padding(.bottom, 8)
    }

    // MARK: - Shimmer

    private func shimmerView() -> some View {
        GeometryReader { geo in
            LinearGradient(
                colors: [.clear, Color.white.opacity(0.3), .clear],
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

        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")

        // Try with fractional seconds first (e.g. "2026-03-26T18:07:19.136")
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSS"
        if let date = formatter.date(from: isoString) {
            let display = DateFormatter()
            display.dateFormat = "'Discovered' MMM d, yyyy"
            return display.string(from: date)
        }

        // Try without fractional seconds (e.g. "2026-03-26T18:07:19")
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
        if let date = formatter.date(from: isoString) {
            let display = DateFormatter()
            display.dateFormat = "'Discovered' MMM d, yyyy"
            return display.string(from: date)
        }

        return "Unknown date"
    }
}

#Preview {
    ZStack {
        // Simulate map behind
        LinearGradient(colors: [.blue.opacity(0.6), .green.opacity(0.6)], startPoint: .top, endPoint: .bottom)
            .ignoresSafeArea()
        VStack { Spacer() }
        PlantDetailSheet(
            plant: Plant(id: 1, latitude: 42.69, longitude: 23.32,
                         imageUrl: nil, userId: 1, createdAt: "2026-03-12T10:00:00"),
            onClose: {}
        )
    }
}
