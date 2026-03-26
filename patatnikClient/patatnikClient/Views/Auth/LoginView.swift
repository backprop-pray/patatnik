import SwiftUI

struct LoginView: View {
    @EnvironmentObject var authVM: AuthViewModel

    @State private var email = ""
    @State private var password = ""
    @State private var appeared = false

    private var isFormValid: Bool {
        !email.isEmpty && !password.isEmpty
    }

    var body: some View {
        ZStack {
            Color(hex: 0x0A0A0F).ignoresSafeArea()

            VStack(spacing: 0) {
                // Robot illustration + branding
                VStack(spacing: 8) {
                    RobotIllustration()

                    Text("TankPlant")
                        .font(.title.bold())
                        .foregroundStyle(.white)

                    Text("Autonomous Plant Intelligence")
                        .font(.subheadline).italic()
                        .foregroundStyle(Color(hex: 0x8E8E93))
                }
                .frame(height: 260)
                .staggerIn(index: 0, appeared: appeared)

                Spacer().frame(height: 24)

                // Form card
                ScrollView {
                    VStack(spacing: 24) {
                        // Error banner
                        if let error = authVM.errorMessage {
                            errorBanner(error)
                                .staggerIn(index: 1, appeared: appeared)
                        }

                        UnderlineField(
                            label: "EMAIL",
                            text: $email,
                            contentType: .emailAddress,
                            keyboardType: .emailAddress
                        )
                        .staggerIn(index: 2, appeared: appeared)

                        UnderlineField(
                            label: "PASSWORD",
                            text: $password,
                            isSecure: true,
                            contentType: .password
                        )
                        .staggerIn(index: 3, appeared: appeared)

                        // Sign In button
                        Button {
                            Task { await authVM.login(email: email, password: password) }
                        } label: {
                            Group {
                                if authVM.isLoading {
                                    ProgressView()
                                        .tint(.white)
                                } else {
                                    Text("Sign In")
                                        .font(.headline)
                                }
                            }
                            .foregroundStyle(.white)
                            .frame(maxWidth: .infinity)
                            .frame(height: 54)
                            .background(
                                LinearGradient(
                                    colors: [Color(hex: 0xFF9500), Color(hex: 0xFF6B00)],
                                    startPoint: .leading,
                                    endPoint: .trailing
                                )
                            )
                            .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                            .shadow(color: Color(hex: 0xFF9500).opacity(0.35), radius: 12, y: 4)
                        }
                        .buttonStyle(ScaleButtonStyle())
                        .disabled(!isFormValid || authVM.isLoading)
                        .opacity(isFormValid && !authVM.isLoading ? 1 : 0.6)
                        .staggerIn(index: 4, appeared: appeared)

                        // Register link
                        HStack(spacing: 4) {
                            Text("Don't have an account?")
                                .font(.subheadline)
                                .foregroundStyle(Color(hex: 0x8E8E93))

                            NavigationLink {
                                RegisterView()
                            } label: {
                                Text("Register")
                                    .font(.subheadline.bold())
                                    .foregroundStyle(Color(hex: 0xFF9500))
                            }
                        }
                        .staggerIn(index: 5, appeared: appeared)
                    }
                    .padding(.horizontal, 28)
                    .padding(.vertical, 32)
                }
                .background(
                    RoundedRectangle(cornerRadius: 28, style: .continuous)
                        .fill(Color(hex: 0x1C1C1E))
                        .shadow(color: .black.opacity(0.4), radius: 24, y: -8)
                )
            }
        }
        .navigationBarBackButtonHidden()
        .onAppear {
            withAnimation { appeared = true }
            authVM.errorMessage = nil
        }
    }

    // MARK: - Error Banner

    private func errorBanner(_ message: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.subheadline)
            Text(message)
                .font(.subheadline)
        }
        .foregroundStyle(Color(hex: 0xFF453A))
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Color(hex: 0xFF453A).opacity(0.12))
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .stroke(Color(hex: 0xFF453A).opacity(0.3), lineWidth: 1)
                )
        )
        .transition(.move(edge: .top).combined(with: .opacity))
    }
}

// MARK: - Scale Button Style

struct ScaleButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.97 : 1.0)
            .animation(.easeInOut(duration: 0.15), value: configuration.isPressed)
    }
}

// MARK: - Stagger Animation Modifier

private struct StaggerModifier: ViewModifier {
    let index: Int
    let appeared: Bool

    func body(content: Content) -> some View {
        content
            .offset(y: appeared ? 0 : 40)
            .opacity(appeared ? 1 : 0)
            .animation(
                .spring(response: 0.5, dampingFraction: 0.8)
                .delay(Double(index) * 0.07),
                value: appeared
            )
    }
}

extension View {
    func staggerIn(index: Int, appeared: Bool) -> some View {
        modifier(StaggerModifier(index: index, appeared: appeared))
    }
}

#Preview {
    NavigationStack {
        LoginView()
            .environmentObject(AuthViewModel())
    }
}
