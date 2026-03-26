package com.patatnik.server.repository;

import com.patatnik.server.model.Plant;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface PlantRepository extends JpaRepository<Plant, Long> {

    List<Plant> findByUserId(Long userId);

    List<Plant> findByUserIdOrderByCreatedAtDesc(Long userId);
}
