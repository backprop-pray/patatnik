package com.patatnik.server.dto;

import com.patatnik.server.model.Plant;
import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class PlantResponse {
    private Long id;
    private Double latitude;
    private Double longitude;
    private String imageUrl;
    private Long userId;
    private String createdAt;

    public static PlantResponse fromEntity(Plant plant) {
        return PlantResponse.builder()
                .id(plant.getId())
                .latitude(plant.getLatitude())
                .longitude(plant.getLongitude())
                .imageUrl(plant.getImageUrl())
                .userId(plant.getUser().getId())
                .createdAt(plant.getCreatedAt() != null ? plant.getCreatedAt().toString() : null)
                .build();
    }
}
