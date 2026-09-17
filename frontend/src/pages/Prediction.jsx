import {
  useState,
  useCallback,
} from "react";

import {
  Play,
  Square,
  Camera,
  Eye,
  Target,
  Brain,
  Activity,
  AlertTriangle,
  Server,
  CheckCircle2,
} from "lucide-react";

import CameraFeed from "../components/CameraFeed";
import GazeVisualizer from "../components/GazeVisualizer";
import HeatmapViewer from "../components/HeatmapViewer";

import {
  predictGaze,
} from "../services/api";


function Prediction() {

  const [monitoring, setMonitoring] =
    useState(false);

  const [driverStream, setDriverStream] =
    useState(null);

  const [sceneStream, setSceneStream] =
    useState(null);

  const [driverCapture, setDriverCapture] =
    useState(null);

  const [sceneCapture, setSceneCapture] =
    useState(null);

  const [loading, setLoading] =
    useState(false);

  const [prediction, setPrediction] =
    useState(null);

  const [error, setError] =
    useState(null);

  const [lastAnalysis, setLastAnalysis] =
    useState(null);


  // ==========================================================
  // START MONITORING
  // ==========================================================

  const startMonitoring = () => {

    setError(null);

    setPrediction(null);

    setLastAnalysis(null);

    setMonitoring(true);

  };


  // ==========================================================
  // STOP MONITORING
  // ==========================================================

  const stopMonitoring = () => {

    setMonitoring(false);

    setDriverStream(null);

    setSceneStream(null);

    setDriverCapture(null);

    setSceneCapture(null);

    setPrediction(null);

    setLastAnalysis(null);

    setError(null);

  };


  // ==========================================================
  // RECEIVE DRIVER STREAM
  // ==========================================================

  const handleDriverStream =
    useCallback(
      (stream) => {

        setDriverStream(stream);

      },
      []
    );


  // ==========================================================
  // RECEIVE SCENE STREAM
  // ==========================================================

  const handleSceneStream =
    useCallback(
      (stream) => {

        setSceneStream(stream);

      },
      []
    );


  // ==========================================================
  // RECEIVE DRIVER CAPTURE FUNCTION
  // ==========================================================

  const handleDriverCapture =
    useCallback(
      (capture) => {

        setDriverCapture(
          () => capture
        );

      },
      []
    );


  // ==========================================================
  // RECEIVE SCENE CAPTURE FUNCTION
  // ==========================================================

  const handleSceneCapture =
    useCallback(
      (capture) => {

        setSceneCapture(
          () => capture
        );

      },
      []
    );


  // ==========================================================
  // RUN REAL L-DRIX-NET INFERENCE
  // ==========================================================

  const analyzeFrame = async () => {

    if (
      !driverStream ||
      !sceneStream
    ) {

      setError(
        "Both cameras must be active."
      );

      return;

    }


    if (
      !driverCapture ||
      !sceneCapture
    ) {

      setError(
        "Camera frames are not ready yet."
      );

      return;

    }


    if (loading) {

      return;

    }


    setLoading(true);

    setError(null);


    try {

      // ------------------------------------------------------
      // Capture current driver frame
      // ------------------------------------------------------

      const driverFrame =
        await driverCapture();


      // ------------------------------------------------------
      // Capture current scene frame
      // ------------------------------------------------------

      const sceneFrame =
        await sceneCapture();


      // ------------------------------------------------------
      // Send frames to FastAPI
      // ------------------------------------------------------

      const result =
        await predictGaze(
          driverFrame,
          sceneFrame
        );


      // ------------------------------------------------------
      // Convert backend result to UI format
      // ------------------------------------------------------

      const gaze =
        result.gaze || {};


      const gazeLocation =
        gaze.gaze_location_2d ||
        [0, 0];


      const sceneWidth =
        result.frames?.scene?.width ||
        942;


      const sceneHeight =
        result.frames?.scene?.height ||
        489;


      const gazeX =
        Number(
          gazeLocation[0] || 0
        );


      const gazeY =
        Number(
          gazeLocation[1] || 0
        );


      const pointX =
        Math.max(
          0,
          Math.min(
            100,
            (gazeX / sceneWidth) * 100
          )
        );


      const pointY =
        Math.max(
          0,
          Math.min(
            100,
            (gazeY / sceneHeight) * 100
          )
        );


      // ------------------------------------------------------
      // Determine viewing direction
      // ------------------------------------------------------

      let direction =
        "CENTER";


      if (pointX < 33) {

        direction = "LEFT";

      } else if (pointX > 67) {

        direction = "RIGHT";

      }


      if (pointY < 30) {

        direction += " / UP";

      } else if (pointY > 70) {

        direction += " / DOWN";

      }


      // ------------------------------------------------------
      // Confidence
      //
      // IMPORTANT:
      // The current backend does not provide a calibrated
      // confidence score for live gaze.
      //
      // We therefore use face detection confidence as the
      // displayed detection confidence rather than pretending
      // it is model gaze confidence.
      // ------------------------------------------------------

      const faceConfidence =
        Number(
          result.face_detection?.confidence ||
          0
        );


      const uiPrediction = {

        ...result,

        direction,

        x: Number(
          gazeX.toFixed(2)
        ),

        y: Number(
          gazeY.toFixed(2)
        ),

        normalizedX: Number(
          (
            gazeX /
            sceneWidth
          ).toFixed(4)
        ),

        normalizedY: Number(
          (
            gazeY /
            sceneHeight
          ).toFixed(4)
        ),

        pointX: Number(
          pointX.toFixed(2)
        ),

        pointY: Number(
          pointY.toFixed(2)
        ),

        confidence: Number(
          (
            faceConfidence * 100
          ).toFixed(1)
        ),

        sceneImage: sceneFrame,

        timestamp:
          new Date().toISOString(),

      };


      setPrediction(
        uiPrediction
      );


      setLastAnalysis(
        new Date()
      );


    } catch (err) {

      console.error(
        "L-DriX-Net prediction error:",
        err
      );


      setError(
        err?.message ||
        "Prediction request failed."
      );

    } finally {

      setLoading(false);

    }

  };


  // ==========================================================
  // CAMERA STATUS
  // ==========================================================

  const camerasReady =
    Boolean(
      driverStream &&
      sceneStream &&
      driverCapture &&
      sceneCapture
    );


  // ==========================================================
  // RENDER
  // ==========================================================

  return (

    <div className="prediction-page">

      {/* ====================================================
          HEADER
      ==================================================== */}

      <section className="prediction-page-header">

        <div>

          <div className="heading-label">

            <Brain size={16} />

            L-DRIX-NET INFERENCE

          </div>


          <h1>
            Live Gaze Prediction
          </h1>


          <p>
            Analyze synchronized driver and
            road-scene camera frames using
            the L-DriX-Net model.
          </p>

        </div>


        <div
          className={`inference-status ${
            monitoring
              ? "running"
              : ""
          }`}
        >

          <span></span>

          {monitoring
            ? "CAMERAS ACTIVE"
            : "SYSTEM READY"}

        </div>

      </section>


      {/* ====================================================
          IMPORTANT INFORMATION
      ==================================================== */}

      <div className="prediction-notice">

        <div className="notice-icon">

          <AlertTriangle
            size={19}
          />

        </div>


        <div>

          <strong>
            Dual-camera input required
          </strong>


          <p>
            Camera 1 observes the driver
            while Camera 2 observes the
            road scene. Both inputs are
            sent to the L-DriX-Net backend
            for multimodal analysis.
          </p>

        </div>

      </div>


      {/* ====================================================
          BACKEND STATUS
      ==================================================== */}

      <div className="backend-status-card">

        <div className="backend-status-icon">

          <Server size={19} />

        </div>


        <div>

          <strong>
            L-DriX-Net Backend
          </strong>

          <span>
            FastAPI • CUDA inference •
            9,006,153 parameters
          </span>

        </div>


        <div className="backend-online">

          <CheckCircle2
            size={16}
          />

          ONLINE

        </div>

      </div>


      {/* ====================================================
          CAMERA INPUTS
      ==================================================== */}

      <section className="prediction-section">

        <div className="prediction-section-header">

          <div>

            <h2>
              Camera Streams
            </h2>

            <p>
              Live input sources
            </p>

          </div>


          <div className="camera-status-summary">

            <Camera size={17} />

            <span>

              {camerasReady
                ? "Both cameras connected"
                : "Waiting for cameras"}

            </span>

          </div>

        </div>


        <div className="prediction-camera-grid">

          <CameraFeed
            cameraType="driver"
            isActive={monitoring}
            onStreamReady={
              handleDriverStream
            }
            onFrameReady={
              handleDriverCapture
            }
          />


          <CameraFeed
            cameraType="scene"
            isActive={monitoring}
            onStreamReady={
              handleSceneStream
            }
            onFrameReady={
              handleSceneCapture
            }
          />

        </div>

      </section>


      {/* ====================================================
          CAMERA CONTROL
      ==================================================== */}

      <section className="prediction-control">

        {!monitoring ? (

          <button
            type="button"
            className="start-monitoring"
            onClick={
              startMonitoring
            }
          >

            <Play
              size={20}
              fill="currentColor"
            />

            Start Camera Monitoring

          </button>

        ) : (

          <button
            type="button"
            className="stop-monitoring"
            onClick={
              stopMonitoring
            }
          >

            <Square
              size={18}
              fill="currentColor"
            />

            Stop Monitoring

          </button>

        )}

      </section>


      {/* ====================================================
          ERROR
      ==================================================== */}

      {error && (

        <div className="prediction-error">

          <AlertTriangle
            size={18}
          />

          <span>
            {error}
          </span>

        </div>

      )}


      {/* ====================================================
          ANALYSIS CONTROL
      ==================================================== */}

      <section className="analysis-control-card">

        <div className="analysis-control-info">

          <div className="analysis-control-icon">

            <Activity
              size={21}
            />

          </div>


          <div>

            <h3>
              Frame Analysis
            </h3>


            <p>

              {loading
                ? "Sending camera frames to L-DriX-Net..."
                : camerasReady
                  ? "Capture the current camera frames and run L-DriX-Net inference."
                  : "Activate both camera streams before running prediction."}

            </p>

          </div>

        </div>


        <button
          type="button"
          className="analyze-button"
          onClick={analyzeFrame}
          disabled={
            !camerasReady ||
            loading
          }
        >

          {loading ? (

            <>

              <span
                className="loading-spinner"
              />

              Processing...

            </>

          ) : (

            <>

              <Eye size={19} />

              Analyze Current Frame

            </>

          )}

        </button>

      </section>


      {/* ====================================================
          LAST ANALYSIS
      ==================================================== */}

      {lastAnalysis && (

        <div className="analysis-complete">

          <CheckCircle2
            size={17}
          />

          <span>
            Last analysis completed at{" "}
            {lastAnalysis.toLocaleTimeString()}
          </span>

        </div>

      )}


      {/* ====================================================
          RESULT AREA
      ==================================================== */}

      <section className="prediction-results">

        {/* ==================================================
            GAZE DIRECTION
        ================================================== */}

        <div className="result-card">

          <div className="result-card-header">

            <div className="result-card-title">

              <div className="result-icon">

                <Eye size={20} />

              </div>


              <div>

                <h3>
                  Gaze Direction
                </h3>


                <p>
                  Current estimated viewing direction
                </p>

              </div>

            </div>

          </div>


          <div className="result-direction">

            {prediction ? (

              <>

                <span>
                  DRIVER IS LOOKING
                </span>


                <strong>
                  {prediction.direction}
                </strong>

              </>

            ) : (

              <>

                <span>
                  WAITING FOR MODEL
                </span>


                <strong>
                  —
                </strong>

              </>

            )}

          </div>

        </div>


        {/* ==================================================
            GAZE POINT
        ================================================== */}

        <div className="result-card">

          <div className="result-card-header">

            <div className="result-card-title">

              <div className="result-icon">

                <Target size={20} />

              </div>


              <div>

                <h3>
                  Gaze Point
                </h3>


                <p>
                  Estimated location in scene
                </p>

              </div>

            </div>

          </div>


          <div className="gaze-coordinates">

            <div>

              <span>
                X Coordinate
              </span>


              <strong>

                {prediction
                  ? `${prediction.x}px`
                  : "—"}

              </strong>

            </div>


            <div>

              <span>
                Y Coordinate
              </span>


              <strong>

                {prediction
                  ? `${prediction.y}px`
                  : "—"}

              </strong>

            </div>


            <div>

              <span>
                Face Detection
              </span>


              <strong>

                {prediction
                  ? `${prediction.confidence}%`
                  : "—"}

              </strong>

            </div>

          </div>

        </div>

      </section>


      {/* ====================================================
          LIVE MODEL DETAILS
      ==================================================== */}

      {prediction && (

        <section className="prediction-results">

          <div className="result-card">

            <div className="result-card-header">

              <div className="result-card-title">

                <div className="result-icon">

                  <Activity size={20} />

                </div>


                <div>

                  <h3>
                    Head Pose
                  </h3>


                  <p>
                    Estimated driver head orientation
                  </p>

                </div>

              </div>

            </div>


            <div className="gaze-coordinates">

              <div>

                <span>
                  Yaw
                </span>


                <strong>
                  {Number(
                    prediction.head_pose?.yaw || 0
                  ).toFixed(1)}°
                </strong>

              </div>


              <div>

                <span>
                  Pitch
                </span>


                <strong>
                  {Number(
                    prediction.head_pose?.pitch || 0
                  ).toFixed(1)}°
                </strong>

              </div>


              <div>

                <span>
                  Roll
                </span>


                <strong>
                  {Number(
                    prediction.head_pose?.roll || 0
                  ).toFixed(1)}°
                </strong>

              </div>

            </div>

          </div>


          <div className="result-card">

            <div className="result-card-header">

              <div className="result-card-title">

                <div className="result-icon">

                  <Brain size={20} />

                </div>


                <div>

                  <h3>
                    Model Output
                  </h3>


                  <p>
                    Actual L-DriX-Net prediction
                  </p>

                </div>

              </div>

            </div>


            <div className="gaze-coordinates">

              <div>

                <span>
                  Prediction
                </span>


                <strong>
                  24-D
                </strong>

              </div>


              <div>

                <span>
                  Heatmap
                </span>


                <strong>
                  224×224
                </strong>

              </div>


              <div>

                <span>
                  Device
                </span>


                <strong>
                  CUDA
                </strong>

              </div>

            </div>

          </div>

        </section>

      )}


      {/* ====================================================
          VISUALIZATION
      ==================================================== */}

      <section className="prediction-visualization">

        <GazeVisualizer
          prediction={prediction}
        />


        <HeatmapViewer
          prediction={prediction}
          sceneImage={
            prediction?.sceneImage || null
          }
        />

      </section>


      {/* ====================================================
          PIPELINE
      ==================================================== */}

      <section className="inference-pipeline">

        <div className="pipeline-heading">

          <h2>
            Inference Pipeline
          </h2>


          <p>
            How the live camera data moves
            through L-DriX-Net.
          </p>

        </div>


        <div className="pipeline">

          <div className="pipeline-step">

            <div className="pipeline-number">
              01
            </div>


            <strong>
              Driver Frame
            </strong>


            <span>
              Face & eye information
            </span>

          </div>


          <div className="pipeline-line"></div>


          <div className="pipeline-step">

            <div className="pipeline-number">
              02
            </div>


            <strong>
              Scene Frame
            </strong>


            <span>
              Road environment
            </span>

          </div>


          <div className="pipeline-line"></div>


          <div className="pipeline-step">

            <div className="pipeline-number">
              03
            </div>


            <strong>
              Feature Extraction
            </strong>


            <span>
              Face + scene encoding
            </span>

          </div>


          <div className="pipeline-line"></div>


          <div className="pipeline-step">

            <div className="pipeline-number">
              04
            </div>


            <strong>
              ICFM Fusion
            </strong>


            <span>
              Multimodal feature fusion
            </span>

          </div>


          <div className="pipeline-line"></div>


          <div className="pipeline-step">

            <div className="pipeline-number">
              05
            </div>


            <strong>
              Gaze Output
            </strong>


            <span>
              Gaze + attention map
            </span>

          </div>

        </div>

      </section>


      {/* ====================================================
          FOOTER
      ==================================================== */}

      <div className="dashboard-footer">

        <div>

          <CheckCircle2
            size={17}
          />

          <span>
            {prediction
              ? "L-DriX-Net analysis complete"
              : "L-DriX-Net system ready"}
          </span>

        </div>


        <span>
          Dual Camera Mode
        </span>

      </div>

    </div>

  );

}


export default Prediction;