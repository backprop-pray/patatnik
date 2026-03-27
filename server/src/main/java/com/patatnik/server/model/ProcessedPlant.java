package com.patatnik.server.model;

import jakarta.persistence.*;
import lombok.*;

import java.time.LocalDateTime;

@Entity
@Table(name = "processed_plants")
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@AllArgsConstructor
@Builder
public class ProcessedPlant {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "plant_id", nullable = false)
    private Plant plant;

    // @Column(nullable = false)
    // private String healthStatus;

    private String disease;

    @Column(nullable = false)
    private String recommendedAction;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "recommended_action_user_id", nullable = false)
    private User recommendedActionUserId;

    @Column(name = "created_at", nullable = false, updatable = false)
    @Builder.Default
    private LocalDateTime createdAt = LocalDateTime.now();

    private Boolean status;

    public void accept() {
        this.status = true;
    }

    public void reject() {
        this.status = false;
    }
}
