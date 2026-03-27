package com.patatnik.server.service.processedPlant;

import com.patatnik.server.model.User;

public interface ProcessedPlantService {
    void accept(Long id, User user);
    void reject(Long id, String comment, User user);
}