package com.patatnik.server.dto;
import com.patatnik.server.model.Plant;
import com.patatnik.server.model.ProcessedPlant;
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
    private String disease;
    private String recommendedAction;
    private Boolean status;
    private Long processPlantId;

    public static PlantResponse fromEntity(Plant plant) {
        ProcessedPlant pp = plant.getProcessedPlant();
        return PlantResponse.builder()
            .id(plant.getId())
            .latitude(plant.getLatitude())
            .longitude(plant.getLongitude())
            .imageUrl(plant.getImageUrl())
            .userId(plant.getUser().getId())
            .createdAt(plant.getCreatedAt() != null
                ? plant.getCreatedAt().toString() : null)
            .disease(pp != null ? pp.getDisease() : null)
            .recommendedAction(pp != null ? pp.getRecommendedAction() : null)
            .status(pp != null ? pp.getStatus() : null)
            .processPlantId(pp != null ? pp.getId(): null)
            .build();
    }
}