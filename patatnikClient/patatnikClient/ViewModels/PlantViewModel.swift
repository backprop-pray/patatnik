import SwiftUI
import Combine

@MainActor
class PlantViewModel: ObservableObject {
    @Published var plants: [Plant] = []
    @Published var selectedPlant: Plant?
    @Published var errorMessage: String?
    @Published var isLoading = false
}
