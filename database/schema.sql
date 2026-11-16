-- =============================================================
-- AquaEnhance — Microsoft SQL Server Schema
-- Purpose : Stores image metadata and user feedback ratings.
-- Note    : No BLOBs — images are stored as file paths.
-- =============================================================

-- ── 1. Create the database (run once; skip if already exists) ──
IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = 'aquaenhance')
BEGIN
    CREATE DATABASE aquaenhance;
END
GO

USE aquaenhance;
GO

-- ── 2. Image table ─────────────────────────────────────────────
-- Stores one row per uploaded image.
-- original_image  : relative path to the uploaded raw file.
-- enhanced_image  : relative path to the model-enhanced file.
-- processing_info : JSON string, e.g. {"patches":64,"inference_ms":1230}
IF NOT EXISTS (
    SELECT * FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_NAME = 'Image'
)
BEGIN
    CREATE TABLE Image (
        image_id        INT             IDENTITY(1,1) PRIMARY KEY,
        original_image  NVARCHAR(512)   NOT NULL,
        enhanced_image  NVARCHAR(512)   NULL,          -- NULL until processing completes
        upload_timestamp DATETIME2      NOT NULL DEFAULT SYSDATETIME(),
        processing_info NVARCHAR(MAX)   NULL           -- JSON string
    );
END
GO

-- ── 3. Feedback table ──────────────────────────────────────────
-- One rating row per image (users can rate after enhancement).
IF NOT EXISTS (
    SELECT * FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_NAME = 'Feedback'
)
BEGIN
    CREATE TABLE Feedback (
        feedback_id INT         IDENTITY(1,1) PRIMARY KEY,
        image_id    INT         NOT NULL,
        rating      TINYINT     NOT NULL CHECK (rating BETWEEN 1 AND 5),
        timestamp   DATETIME2   NOT NULL DEFAULT SYSDATETIME(),

        CONSTRAINT FK_Feedback_Image
            FOREIGN KEY (image_id) REFERENCES Image(image_id)
            ON DELETE CASCADE
    );
END
GO

-- ── 4. Index for fast feedback look-up by image ────────────────
IF NOT EXISTS (
    SELECT * FROM sys.indexes
    WHERE name = 'IX_Feedback_image_id'
)
BEGIN
    CREATE INDEX IX_Feedback_image_id ON Feedback(image_id);
END
GO
