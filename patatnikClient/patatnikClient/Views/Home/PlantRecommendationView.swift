import SwiftUI

struct PlantRecommendationView: View {
    @ObservedObject var viewModel: PlantDetailViewModel

    var body: some View {
        Group {
            if viewModel.isLoadingRecommendation && !viewModel.recommendationError {
                loadingState
            } else if viewModel.recommendationLoaded {
                loadedState
            } else if viewModel.recommendationError {
                errorState
            }
        }
    }

    // MARK: - Loading State

    private var loadingState: some View {
        VStack(spacing: 0) {
            // Thin animated progress bar across the top
            AnalysisProgressBar()
                .frame(height: 3)
                .padding(.horizontal, 20)
                .padding(.top, 16)

            // Robot — with breathing room
            RobotLoadingView()
                .frame(width: 140, height: 160)
                .padding(.top, 16)

            // Text labels below robot
            VStack(spacing: 4) {
                Text("Analyzing Plant Health")
                    .font(.system(size: 15, weight: .medium))
                    .foregroundStyle(.primary)

                HStack(spacing: 4) {
                    Image(systemName: "sparkles")
                        .font(.system(size: 11))
                        .foregroundStyle(Color.appOrange)
                    Text("Powered by TankPlant AI")
                        .font(.system(size: 12, weight: .regular))
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.top, 8)
            .padding(.bottom, 24)
        }
        .frame(maxWidth: .infinity)
        .frame(minHeight: 220)
        .background(
            // Warm gradient background — orange 4% at top fading to clear
            LinearGradient(
                colors: [Color.appOrange.opacity(0.04), .clear],
                startPoint: .top,
                endPoint: .bottom
            )
        )
        .padding(.horizontal, 20)
        .padding(.vertical, 16)
    }

    // MARK: - Loaded State

    private var loadedState: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Block 1 — Diagnosis card
            VStack(alignment: .leading, spacing: 8) {
                Text("DETECTED CONDITION")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .kerning(0.8)

                HStack(spacing: 8) {
                    Image(systemName: "staroflife.fill")
                        .font(.system(size: 14, weight: .bold))
                        .foregroundStyle(.white)

                    Text(viewModel.estimatedDisease)
                        .font(.system(size: 17, weight: .bold))
                        .foregroundStyle(.white)
                        .lineLimit(2)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 12)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(
                    RoundedRectangle(cornerRadius: 14)
                        .fill(
                            LinearGradient(
                                colors: [Color.red, Color.red.opacity(0.8)],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .shadow(color: Color.red.opacity(0.3), radius: 8, y: 4)
                )
            }
            .transition(.scale(scale: 0.85).combined(with: .opacity))

            // Block 2 — Treatment card
            VStack(alignment: .leading, spacing: 8) {
                Text("RECOMMENDED ACTION")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .kerning(0.8)

                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "cross.circle.fill")
                        .font(.system(size: 14))
                        .foregroundStyle(Color.appOrange)
                        .padding(.top, 2)

                    Text(viewModel.recommendation)
                        .font(.system(size: 15, weight: .regular))
                        .foregroundStyle(.primary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(16)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(
                    RoundedRectangle(cornerRadius: 14)
                        .fill(Color.appOrange.opacity(0.06))
                        .overlay(
                            RoundedRectangle(cornerRadius: 14)
                                .stroke(Color.appOrange.opacity(0.18), lineWidth: 1)
                        )
                )
            }
            .transition(.move(edge: .bottom).combined(with: .opacity))

            // Action prompt
            Text("Does this look right?")
                .font(.system(size: 12, weight: .regular))
                .foregroundStyle(.secondary)
                .padding(.top, 4)

            // Accept / Reject buttons — taller, more confident
            HStack(spacing: 12) {
                // REJECT
                Button { viewModel.respond(accepted: false) } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "xmark")
                        Text("Reject")
                    }
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(.red)
                    .frame(maxWidth: .infinity)
                    .frame(height: 54)
                    .background(
                        RoundedRectangle(cornerRadius: 14)
                            .stroke(Color.red.opacity(0.35), lineWidth: 1.5)
                    )
                }
                .buttonStyle(RecommendationButtonStyle())

                // ACCEPT
                Button { viewModel.respond(accepted: true) } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "checkmark")
                        Text("Accept")
                    }
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(.white)
                    .frame(maxWidth: .infinity)
                    .frame(height: 54)
                    .background(
                        RoundedRectangle(cornerRadius: 14)
                            .fill(Color.appOrange)
                            .shadow(color: Color.appOrange.opacity(0.35),
                                    radius: 10, y: 4)
                    )
                }
                .buttonStyle(RecommendationButtonStyle())
            }
            .disabled(viewModel.hasResponded)
            .opacity(viewModel.hasResponded ? 0.5 : 1.0)
            .animation(.easeInOut(duration: 0.2), value: viewModel.hasResponded)

            // Response confirmation
            if viewModel.hasResponded {
                HStack(spacing: 6) {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(Color.appOrange)
                    Text("Response recorded")
                        .font(.system(size: 12, weight: .regular))
                        .foregroundStyle(.secondary)
                }
                .transition(.opacity.animation(.easeIn))
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 16)
    }

    // MARK: - Error State

    private var errorState: some View {
        HStack(spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 18))
                .foregroundStyle(.orange)
            VStack(alignment: .leading, spacing: 2) {
                Text("Analysis unavailable")
                    .font(.system(size: 15, weight: .medium))
                    .foregroundStyle(.primary)
                Text("Could not load recommendation")
                    .font(.system(size: 12, weight: .regular))
                    .foregroundStyle(.secondary)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 14)
                .fill(Color.orange.opacity(0.06))
                .overlay(
                    RoundedRectangle(cornerRadius: 14)
                        .stroke(Color.orange.opacity(0.15), lineWidth: 1)
                )
        )
        .padding(.horizontal, 20)
        .padding(.vertical, 16)
    }
}

// MARK: - Analysis Progress Bar

private struct AnalysisProgressBar: View {
    @State private var offset: CGFloat = -1.0

    var body: some View {
        GeometryReader { geo in
            let barWidth = geo.size.width * 0.35

            RoundedRectangle(cornerRadius: 1.5)
                .fill(Color.appOrange.opacity(0.1))
                .overlay(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 1.5)
                        .fill(
                            LinearGradient(
                                colors: [Color.appOrange.opacity(0.3), Color.appOrange, Color.appOrange.opacity(0.3)],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .frame(width: barWidth)
                        .offset(x: offset * (geo.size.width + barWidth) - barWidth)
                }
                .clipShape(RoundedRectangle(cornerRadius: 1.5))
        }
        .onAppear {
            withAnimation(.linear(duration: 2.0).repeatForever(autoreverses: false)) {
                offset = 1.0
            }
        }
    }
}

// MARK: - Recommendation Button Style

private struct RecommendationButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.96 : 1.0)
            .animation(.easeInOut(duration: 0.15), value: configuration.isPressed)
    }
}

// MARK: - Opinion Input View

struct OpinionInputView: View {
    @ObservedObject var viewModel: PlantDetailViewModel
    let plantId: Int
    @FocusState private var isFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            // Section header
            HStack(spacing: 6) {
                Image(systemName: "pencil.line")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(Color.appOrange)
                Text("Your Diagnosis")
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(.primary)
            }

            Text("How would you treat this plant?")
                .font(.system(size: 12, weight: .regular))
                .foregroundStyle(.secondary)

            if viewModel.opinionSubmitted {
                // Success card — replaces the input
                HStack(spacing: 8) {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 20))
                        .foregroundStyle(.green)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Saved")
                            .font(.system(size: 15, weight: .semibold))
                            .foregroundStyle(.primary)
                        Text("Your diagnosis has been recorded")
                            .font(.system(size: 12, weight: .regular))
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(16)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(
                    RoundedRectangle(cornerRadius: 14)
                        .fill(Color.green.opacity(0.06))
                        .overlay(
                            RoundedRectangle(cornerRadius: 14)
                                .stroke(Color.green.opacity(0.15), lineWidth: 1)
                        )
                )
                .transition(.opacity.combined(with: .scale(scale: 0.95)).animation(.easeInOut(duration: 0.3)))
            } else {
                // Text input
                ZStack(alignment: .topLeading) {
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color.appSurface.opacity(0.04))
                        .overlay(
                            RoundedRectangle(cornerRadius: 12)
                                .stroke(
                                    isFocused
                                        ? Color.appOrange.opacity(0.6)
                                        : Color.appSecondary.opacity(0.15),
                                    lineWidth: 1
                                )
                                .animation(.easeInOut(duration: 0.2), value: isFocused)
                        )
                        .frame(height: 90)

                    TextEditor(text: $viewModel.userOpinion)
                        .font(.system(size: 15, weight: .regular))
                        .foregroundStyle(.primary)
                        .padding(10)
                        .frame(height: 90)
                        .background(.clear)
                        .scrollContentBackground(.hidden)
                        .focused($isFocused)

                    if viewModel.userOpinion.isEmpty {
                        Text("e.g. Apply fungicide spray weekly...")
                            .font(.system(size: 15, weight: .regular))
                            .foregroundStyle(.secondary.opacity(0.6))
                            .padding(14)
                            .allowsHitTesting(false)
                    }
                }
                .transition(.opacity.combined(with: .scale(scale: 0.95)).animation(.easeInOut(duration: 0.3)))

                // Submit button
                Button {
                    isFocused = false
                    Task { await viewModel.submitOpinion(plantId: plantId) }
                } label: {
                    Group {
                        if viewModel.isSubmittingOpinion {
                            ProgressView().tint(.white)
                        } else {
                            HStack(spacing: 8) {
                                Image(systemName: "paperplane.fill")
                                Text("Submit Diagnosis")
                            }
                        }
                    }
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(.white)
                    .frame(maxWidth: .infinity)
                    .frame(height: 50)
                    .background(
                        RoundedRectangle(cornerRadius: 14)
                            .fill(
                                viewModel.userOpinion.trimmingCharacters(
                                    in: .whitespaces).isEmpty
                                    ? Color.appSecondary
                                    : Color.appDark
                            )
                    )
                }
                .buttonStyle(OpinionSubmitButtonStyle())
                .disabled(
                    viewModel.userOpinion.trimmingCharacters(in: .whitespaces).isEmpty
                    || viewModel.isSubmittingOpinion
                    || viewModel.opinionSubmitted
                )
                .animation(.easeInOut(duration: 0.2), value: viewModel.userOpinion)
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 16)
        .animation(.easeInOut(duration: 0.3), value: viewModel.opinionSubmitted)
    }
}

// MARK: - Opinion Submit Button Style

private struct OpinionSubmitButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.97 : 1.0)
            .animation(.easeInOut(duration: 0.15), value: configuration.isPressed)
    }
}
