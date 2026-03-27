import SwiftUI
import MapKit

struct HomeView: View {
    @EnvironmentObject var authVM: AuthViewModel
    @EnvironmentObject var plantVM: PlantViewModel
    @State private var mapType: MKMapType = .hybridFlyover
    @State private var showError = false

    var body: some View {
        ZStack {

            MapView(
                plants: plantVM.plants,
                mapType: mapType,
                selectedPlant: $plantVM.selectedPlant
            )
            .ignoresSafeArea()

            if plantVM.isLoading && plantVM.isFirstLoad {
                ProgressView("Loading plants...")
                    .progressViewStyle(.circular)
                    .padding(24)
                    .background(.ultraThinMaterial)
                    .cornerRadius(16)
                    .shadow(radius: 8)
            }

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

            // Plant detail sheet overlay
            if let plant = plantVM.selectedPlant {
                Color.black.opacity(0.3)
                    .ignoresSafeArea()
                    .onTapGesture {
                        withAnimation(.spring(response: 0.35)) {
                            plantVM.selectedPlant = nil
                        }
                    }
                    .transition(.opacity)
                    .zIndex(199)

                VStack {
                    Spacer()
                    PlantDetailSheet(
                        plant: plant,
                        onClose: {
                            withAnimation(.spring(response: 0.35)) {
                                plantVM.selectedPlant = nil
                            }
                        }
                    )
                }
                .ignoresSafeArea(edges: .bottom)
                .transition(.move(edge: .bottom).combined(with: .opacity))
                .zIndex(200)
            }
        }
        .animation(.spring(response: 0.4, dampingFraction: 0.82),
                    value: plantVM.selectedPlant?.id)
        .task {
            plantVM.bindWebSocket()

            await plantVM.loadPlants(token: authVM.token)

            if let userId = authVM.currentUser?.id {
                PlantWebSocketService.shared.connect(userId: userId)
            }
        }
        .onAppear {
            // Refresh plants when view appears to get latest data
            Task {
                await plantVM.loadPlants(token: authVM.token)
            }
        }
        .onDisappear {
            plantVM.stopListening()
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
