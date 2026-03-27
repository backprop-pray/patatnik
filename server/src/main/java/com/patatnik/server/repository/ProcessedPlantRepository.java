package com.patatnik.server.repository;

import com.patatnik.server.model.ProcessedPlant;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.transaction.annotation.Transactional;

import java.util.Optional;

public interface ProcessedPlantRepository extends JpaRepository<ProcessedPlant, Long> {

    Optional<ProcessedPlant> findById(Long id);

    @Modifying
    @Transactional
    @Query("""
        UPDATE ProcessedPlant pp
        SET pp.recommendedAction = :comment
        WHERE pp.disease = :disease
    """)
    void updateRecommendedActionByDisease(
        @Param("disease") String disease,
        @Param("comment") String comment
    );
}