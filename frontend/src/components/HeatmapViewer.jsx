import {
  Flame,
  Maximize2,
  Eye,
  Info,
} from "lucide-react";

function HeatmapViewer({ prediction, sceneImage }) {
  const hasHeatmap = Boolean(prediction);

  return (
    <div className="heatmap-card">

      {/* Header */}
      <div className="heatmap-header">

        <div className="heatmap-title">

          <div className="heatmap-icon">
            <Flame size={21} />
          </div>

          <div>
            <h3>Driver Attention Heatmap</h3>

            <p>
              Spatial visualization of predicted gaze
              attention
            </p>
          </div>

        </div>

        <button
          type="button"
          className="heatmap-expand"
          title="Expand heatmap"
        >
          <Maximize2 size={18} />
        </button>

      </div>

      {/* Heatmap Display */}
      <div className="heatmap-display">

        {hasHeatmap ? (

          <div className="heatmap-image-container">

            {/* 
              Scene image will be displayed here when
              the backend starts returning the actual
              scene frame.
            */}

            {sceneImage?.url ? (
              <img
                src={sceneImage.url}
                alt="Road scene"
                className="heatmap-background"
              />
            ) : (
              <div className="heatmap-placeholder-background">
                <Eye size={45} />
                <span>Scene Frame</span>
              </div>
            )}

            {/* Simulated attention regions */}
            <div className="attention-region region-one"></div>

            <div className="attention-region region-two"></div>

            <div className="attention-region region-three"></div>

            {/* Predicted gaze point */}
            <div
              className="heatmap-gaze-point"
              style={{
                left: `${prediction.pointX || 50}%`,
                top: `${prediction.pointY || 50}%`,
              }}
            >
              <span></span>
            </div>

            {/* Gaze label */}
            <div className="heatmap-point-label">
              Gaze Point
            </div>

            {/* Overlay */}
            <div className="heatmap-overlay"></div>

            {/* LIVE label */}
            <div className="heatmap-live">
              <span></span>
              ATTENTION MAP
            </div>

          </div>

        ) : (

          <div className="heatmap-empty">

            <div className="empty-heatmap-icon">
              <Flame size={42} />
            </div>

            <h3>No attention map available</h3>

            <p>
              Start live monitoring to generate the
              driver's attention heatmap.
            </p>

          </div>

        )}

      </div>

      {/* Legend */}
      <div className="heatmap-footer">

        <div className="heatmap-legend">

          <span>Low Attention</span>

          <div className="legend-bar"></div>

          <span>High Attention</span>

        </div>

        <div className="heatmap-info">

          <Info size={15} />

          <span>
            Heatmap represents predicted visual
            attention
          </span>

        </div>

      </div>

    </div>
  );
}

export default HeatmapViewer;