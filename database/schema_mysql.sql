-- =============================================================
-- AquaEnhance — MySQL 8.x Schema  (docker-compose fallback)
-- =============================================================

CREATE DATABASE IF NOT EXISTS aquaenhance
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE aquaenhance;

-- ── Image table ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS Image (
    image_id         INT            AUTO_INCREMENT PRIMARY KEY,
    original_image   VARCHAR(512)   NOT NULL,
    enhanced_image   VARCHAR(512)   NULL,
    upload_timestamp DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    processing_info  JSON           NULL
) ENGINE=InnoDB;

-- ── Feedback table ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS Feedback (
    feedback_id  INT       AUTO_INCREMENT PRIMARY KEY,
    image_id     INT       NOT NULL,
    rating       TINYINT   NOT NULL CHECK (rating BETWEEN 1 AND 5),
    timestamp    DATETIME  NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT FK_Feedback_Image
        FOREIGN KEY (image_id) REFERENCES Image(image_id)
        ON DELETE CASCADE
) ENGINE=InnoDB;

CREATE INDEX IF NOT EXISTS IX_Feedback_image_id ON Feedback(image_id);
