import SwiftUI
import MapKit

struct HomeView: View {
    @EnvironmentObject var authVM: AuthViewModel
    @EnvironmentObject var plantVM: PlantViewModel
    @State private var mapType: MKMapType = .hybridFlyover
    @State private var showError = false

    var body: some View {
        ZStack {

            // 1. Full screen map
            MapView(
                plants: plantVM.plants,
                mapType: mapType,
                selectedPlant: $plantVM.selectedPlant
            )
            .ignoresSafeArea()

            // 2. Loading overlay
            if plantVM.isLoading {
                ProgressView("Loading plants...")
                    .progressViewStyle(.circular)
                    .padding(24)
                    .background(.ultraThinMaterial)
                    .cornerRadius(16)
                    .shadow(radius: 8)
            }

            // 3. Error toast
            if showError, let message = plantVM.errorMessage {
                VStack {
                    HStack(spacing: 8) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.white)
                        Text(message)
                            .font(.subheadline.weight(.medium))
                            .foregroundStyle(.white)
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .background(Color.red.opacity(0.9))
                    .clipShape(Capsule())
                    .shadow(radius: 6)
                    .padding(.top, 60)
                    Spacer()
                }
                .transition(.move(edge: .top).combined(with: .opacity))
            }

            // 4. Map type toggle
            VStack {
                Spacer()
                HStack {
                    Spacer()
                    Button {
                        withAnimation(.easeInOut(duration: 0.3)) {
                            switch mapType {
                            case .hybridFlyover: mapType = .hybrid
                            case .hybrid:        mapType = .standard
                            default:             mapType = .hybridFlyover
                            }
                        }
                    } label: {
                        Image(systemName: mapTypeIcon)
                            .font(.system(size: 18, weight: .semibold))
                            .foregroundStyle(.primary)
                            .frame(width: 44, height: 44)
                            .background(.regularMaterial)
                            .clipShape(Circle())
                            .shadow(color: .black.opacity(0.2), radius: 6, x: 0, y: 2)
                    }
                    .padding(.trailing, 16)
                    .padding(.bottom, 40)
                }
            }
        }
        .task {
            await plantVM.loadPlants()
        }
        .onChange(of: plantVM.errorMessage) { _, newValue in
            if newValue != nil {
                withAnimation { showError = true }
                DispatchQueue.main.asyncAfter(deadline: .now() + 4) {
                    withAnimation {
                        showError = false
                        plantVM.errorMessage = nil
                    }
                }
            }
        }
    }

    private var mapTypeIcon: String {
        switch mapType {
        case .hybridFlyover: return "mountain.2.fill"
        case .hybrid:        return "globe.americas.fill"
        default:             return "map.fill"
        }
    }
}

#Preview {
    HomeView()
        .environmentObject(AuthViewModel())
        .environmentObject(PlantViewModel())
}
