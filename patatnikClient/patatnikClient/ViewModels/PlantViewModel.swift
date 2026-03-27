import SwiftUI
import Combine

@MainActor
class PlantViewModel: ObservableObject {
    @Published var plants: [Plant] = []
    @Published var selectedPlant: Plant?
    @Published var errorMessage: String?
    @Published var isLoading = false
    @Published var isFirstLoad = true

    private let plantService = PlantService()


    func loadPlants(token: String?) async {
        guard let token, !token.isEmpty else {
            errorMessage = "Not authenticated."
            return
        }
        guard !isLoading else { return }
        if isFirstLoad { isLoading = true }
        errorMessage = nil
        defer {
            isLoading = false
            isFirstLoad = false
        }

        do {
            plants = try await plantService.getPlants(token: token)
            print("Loaded \(plants.count) plants from API")
        } catch let error as PlantError {
            errorMessage = error.errorDescription
        } catch {
            print("PlantViewModel fetch error:", error)
            errorMessage = "Failed to load plants."
        }
    }


    func createPlant(token: String?, latitude: Double, longitude: Double, image: Data?) async {
        guard let token, !token.isEmpty else {
            errorMessage = "Not authenticated."
            return
        }

        do {
            let newPlant = try await plantService.createPlant(
                token: token,
                latitude: latitude,
                longitude: longitude,
                image: image
            )
            plants.insert(newPlant, at: 0)
            print("Created plant: id=\(newPlant.id)")
        } catch let error as PlantError {
            errorMessage = error.errorDescription
        } catch {
            print("PlantViewModel create error:", error)
            errorMessage = "Failed to create plant."
        }
    }


    func bindWebSocket() {
        print("[PlantVM] Binding onPlantReceived callback")
        PlantWebSocketService.shared.onPlantReceived = { [weak self] plant in
            guard let self else { return }
            
            // Check if plant already exists
            if let existingIndex = self.plants.firstIndex(where: { $0.id == plant.id }) {
                // Update existing plant with new data (includes disease, recommendedAction, status)
                self.plants[existingIndex] = plant
                print("[PlantVM] Plant updated: id=\(plant.id)")
            } else {
                // Add new plant
                self.plants.insert(plant, at: 0)
                print("[PlantVM] Plant added to map: id=\(plant.id)")
            }
        }
    }

    func startListening(userId: Int) {
        bindWebSocket()
        PlantWebSocketService.shared.connect(userId: userId)
    }

    func stopListening() {
        PlantWebSocketService.shared.onPlantReceived = nil
    }
}
