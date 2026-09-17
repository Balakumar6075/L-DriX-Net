import {
  ScanEye,
  Play,
  RotateCcw,
  CheckCircle2,
  LoaderCircle,
  Target,
  Gauge,
} from "lucide-react";


function PredictionPanel({
  prediction,
  loading = false,
  onPredict,
  onReset,
  camerasReady = false,
}) {

  const hasPrediction =
    Boolean(prediction);


  // ==========================================================
  // REAL PREDICTION DATA
  // ==========================================================

  const direction =
    prediction?.direction || "--";


  const gazeX =
    prediction?.x !== undefined &&
    Number.isFinite(Number(prediction.x))
      ? Number(prediction.x).toFixed(2)
      : "--";


  const gazeY =
    prediction?.y !== undefined &&
    Number.isFinite(Number(prediction.y))
      ? Number(prediction.y).toFixed(2)
      : "--";


  // ----------------------------------------------------------
  // IMPORTANT:
  // This is FACE DETECTION confidence,
  // NOT L-DriX-Net prediction confidence.
  // ----------------------------------------------------------

  const faceConfidence =
    prediction?.faceConfidence !== undefined
      ? `${(
          Number(prediction.faceConfidence) * 100
        ).toFixed(1)}%`
      : "--";


  return (

    <div className="prediction-card">

      {/* =====================================================
          HEADER
      ====================================================== */}

      <div className="prediction-header">

        <div className="prediction-title">

          <div className="prediction-icon">

            <ScanEye size={21} />

          </div>


          <div>

            <h3>
              Gaze Prediction
            </h3>

            <p>
              L-DriX-Net driver gaze estimation
            </p>

          </div>

        </div>


        <div
          className={`prediction-status ${
            hasPrediction
              ? "detected"
              : loading
              ? "processing"
              : "waiting"
          }`}
        >

          <span></span>

          {hasPrediction
            ? "DETECTED"
            : loading
            ? "PROCESSING"
            : "WAITING"}

        </div>

      </div>


      {/* =====================================================
          MAIN RESULT
      ====================================================== */}

      <div className="prediction-result">

        <div className="result-main">

          <div className="result-label">

            <Target size={17} />

            Predicted Direction

          </div>


          <div className="result-direction">

            {loading ? (

              <LoaderCircle
                size={28}
                className="rotate-icon"
              />

            ) : (

              direction

            )}

          </div>


          <p className="result-description">

            {hasPrediction

              ? "Current driver viewing direction"

              : camerasReady

              ? "Ready to analyze the current frame"

              : "Activate both cameras to begin analysis"}

          </p>

        </div>


        {/* =================================================
            FACE DETECTION CONFIDENCE
        ================================================== */}

        <div className="confidence-box">

          <div className="confidence-icon">

            <Gauge size={18} />

          </div>


          <div>

            <span>
              Face Detection
            </span>

            <strong>
              {faceConfidence}
            </strong>

          </div>

        </div>

      </div>


      {/* =====================================================
          GAZE COORDINATES
      ====================================================== */}

      <div className="prediction-values">

        <div className="prediction-value">

          <span>
            Gaze X
          </span>

          <strong>
            {gazeX}
          </strong>

        </div>


        <div className="prediction-value">

          <span>
            Gaze Y
          </span>

          <strong>
            {gazeY}
          </strong>

        </div>


        <div className="prediction-value">

          <span>
            Model
          </span>

          <strong>
            {prediction?.modelName ||
              "L-DriX-Net"}
          </strong>

        </div>

      </div>


      {/* =====================================================
          CAMERA STATUS
      ====================================================== */}

      <div className="prediction-input-status">

        <div
          className={`input-status-item ${
            camerasReady ? "ready" : ""
          }`}
        >

          <span className="input-status-dot"></span>


          <div>

            <strong>
              Driver Camera
            </strong>

            <span>

              {camerasReady
                ? "Frame ready"
                : "Waiting"}

            </span>

          </div>

        </div>


        <div
          className={`input-status-item ${
            camerasReady ? "ready" : ""
          }`}
        >

          <span className="input-status-dot"></span>


          <div>

            <strong>
              Scene Camera
            </strong>

            <span>

              {camerasReady
                ? "Frame ready"
                : "Waiting"}

            </span>

          </div>

        </div>

      </div>


      {/* =====================================================
          ACTION BUTTONS
      ====================================================== */}

      <div className="prediction-actions">

        {!hasPrediction ? (

          <button
            type="button"
            className="prediction-run-button"
            onClick={onPredict}
            disabled={
              !camerasReady ||
              loading
            }
          >

            {loading ? (

              <>

                <LoaderCircle
                  size={18}
                  className="rotate-icon"
                />

                Analyzing...

              </>

            ) : (

              <>

                <Play
                  size={18}
                  fill="currentColor"
                />

                Run Prediction

              </>

            )}

          </button>

        ) : (

          <button
            type="button"
            className="prediction-reset-button"
            onClick={onReset}
          >

            <RotateCcw size={17} />

            Reset Prediction

          </button>

        )}

      </div>


      {/* =====================================================
          SUCCESS MESSAGE
      ====================================================== */}

      {hasPrediction && (

        <div className="prediction-success">

          <CheckCircle2 size={17} />


          <div>

            <strong>
              Prediction completed
            </strong>


            <span>
              L-DriX-Net successfully analyzed
              the current camera frames.
            </span>

          </div>

        </div>

      )}

    </div>

  );

}


export default PredictionPanel;