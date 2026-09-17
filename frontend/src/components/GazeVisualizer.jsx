import {
  Eye,
  ArrowUp,
  ArrowDown,
  ArrowLeft,
  ArrowRight,
} from "lucide-react";

function GazeVisualizer({ prediction }) {
  const direction = prediction?.direction || "CENTER";

  return (
    <div className="gaze-card">

      <div className="gaze-card-header">

        <div className="gaze-title">

          <div className="gaze-icon">
            <Eye size={21} />
          </div>

          <div>
            <h3>Gaze Direction</h3>

            <p>
              Predicted driver viewing direction
            </p>
          </div>

        </div>

        <div className="gaze-status">
          {prediction ? "DETECTED" : "WAITING"}
        </div>

      </div>

      <div className="gaze-display">

        <div className="gaze-label top">
          <ArrowUp size={18} />
          UP
        </div>

        <div className="gaze-label left">
          <ArrowLeft size={18} />
          LEFT
        </div>

        <div className="gaze-label right">
          RIGHT
          <ArrowRight size={18} />
        </div>

        <div className="gaze-label bottom">
          <ArrowDown size={18} />
          DOWN
        </div>

        <div className="gaze-crosshair">

          <div className="crosshair-horizontal"></div>
          <div className="crosshair-vertical"></div>

          <div
            className={`gaze-point ${
              prediction ? "detected" : ""
            }`}
          ></div>

        </div>

      </div>

      <div className="gaze-result">

        <span>Current Direction</span>

        <strong>{direction}</strong>

      </div>

    </div>
  );
}

export default GazeVisualizer;