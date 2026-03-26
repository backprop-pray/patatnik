import SwiftUI
import Combine

@MainActor
class AuthViewModel: ObservableObject {
    @Published var isAuthenticated = false
    @Published var currentUser: User?
    @Published var errorMessage: String?
    @Published var isLoading = false

    private let authService = AuthService()

    func login(email: String, password: String) async {
        errorMessage = nil
        isLoading = true
        defer { isLoading = false }

        do {
            let response = try await authService.login(email: email, password: password)
            currentUser = response.user
            isAuthenticated = true
        } catch let error as AuthService.AuthError {
            errorMessage = error.userMessage
        } catch {
            errorMessage = "Something went wrong. Please try again."
        }
    }

    func register(name: String, email: String, password: String) async {
        errorMessage = nil
        isLoading = true
        defer { isLoading = false }

        do {
            let response = try await authService.register(name: name, email: email, password: password)
            currentUser = response.user
            isAuthenticated = true
        } catch let error as AuthService.AuthError {
            errorMessage = error.userMessage
        } catch {
            errorMessage = "Something went wrong. Please try again."
        }
    }

    func logout() {
        currentUser = nil
        isAuthenticated = false
    }
}
