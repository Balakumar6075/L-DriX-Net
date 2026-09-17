import { useEffect, useRef, useState } from "react";

import {
  Activity,
  AlertCircle,
  Brain,
  Camera,
  CheckCircle2,
  Eye,
  Gauge,
  Monitor,
  Play,
  RefreshCw,
  Square,
  Target,
  Wifi,
  WifiOff,
} from "lucide-react";

import {
  predictGaze,
  checkBackendStatus,
} from "../services/api";


// ============================================================
// CONFIGURATION
// ============================================================

const API_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const TEMPORAL_LENGTH = 16;

// Delay between completed inference requests.
// We deliberately wait after each response instead of using
// overlapping requests.
const INFERENCE_INTERVAL = 250;


// ============================================================
// DRIVER STATE ENGINE
// ============================================================
//
// This is the CURRENT PROTOTYPE state engine.
//
// It is NOT the future trained 3-class neural classifier.
//
// It uses the same temporal idea we tested with the backend:
//   - head pose
//   - gaze position
//   - temporal variation
//
// Later this can be replaced by the trained state_head.
// ============================================================

class PrototypeStateEngine {

  constructor(windowSize = TEMPORAL_LENGTH) {
    this.windowSize = windowSize;
    this.history = [];
  }

  reset() {
    this.history = [];
  }

  safeNumber(value, fallback = 0) {
    const number = Number(value);

    if (!Number.isFinite(number)) {
      return fallback;
    }

    return number;
  }

  mean(values) {
    if (!values.length) {
      return 0;
    }

    return (
      values.reduce(
        (sum, value) => sum + value,
        0
      ) / values.length
    );
  }

  std(values) {
    if (!values.length) {
      return 0;
    }

    const average = this.mean(values);

    const variance =
      this.mean(
        values.map(
          (value) =>
            Math.pow(value - average, 2)
        )
      );

    return Math.sqrt(variance);
  }

  range(values) {
    if (!values.length) {
      return 0;
    }

    return (
      Math.max(...values) -
      Math.min(...values)
    );
  }

  extractFrameFeatures(result) {

    const head =
      result?.head_pose ||
      result?.headPose ||
      {};

    const gaze =
      result?.gaze ||
      {};

    const gazeLocation =
      result?.gaze_location_2d ||
      gaze?.gaze_location_2d ||
      [0, 0];

    return {

      yaw: this.safeNumber(
        head?.yaw
      ),

      pitch: this.safeNumber(
        head?.pitch
      ),

      roll: this.safeNumber(
        head?.roll
      ),

      gazeX: this.safeNumber(
        gazeLocation?.[0]
      ),

      gazeY: this.safeNumber(
        gazeLocation?.[1]
      ),
    };
  }

  calculateFeatures() {

    const yaw = this.history.map(
      (item) => item.yaw
    );

    const pitch = this.history.map(
      (item) => item.pitch
    );

    const roll = this.history.map(
      (item) => item.roll
    );

    const gazeX = this.history.map(
      (item) => item.gazeX
    );

    const gazeY = this.history.map(
      (item) => item.gazeY
    );

    return {

      yawMean: this.mean(yaw),
      yawStd: this.std(yaw),
      yawRange: this.range(yaw),

      pitchMean: this.mean(pitch),
      pitchStd: this.std(pitch),
      pitchRange: this.range(pitch),

      rollMean: this.mean(roll),
      rollStd: this.std(roll),
      rollRange: this.range(roll),

      gazeXMean: this.mean(gazeX),
      gazeYMean: this.mean(gazeY),

      gazeXStd: this.std(gazeX),
      gazeYStd: this.std(gazeY),

      gazeXRange: this.range(gazeX),
      gazeYRange: this.range(gazeY),
    };
  }

  update(result) {

    const frameFeatures =
      this.extractFrameFeatures(result);

    this.history.push(
      frameFeatures
    );

    if (
      this.history.length >
      this.windowSize
    ) {
      this.history.shift();
    }

    // --------------------------------------------------------
    // Temporal calibration
    // --------------------------------------------------------

    if (
      this.history.length <
      this.windowSize
    ) {

      return {
        state: "CALIBRATING",
        confidence: 0,
        scores: {
          CONCENTRATED: 0,
          DISTRACTED: 0,
          DROWSY: 0,
        },
        frames:
          this.history.length,
        requiredFrames:
          this.windowSize,
      };
    }

    const features =
      this.calculateFeatures();

    // ========================================================
    // DROWSINESS
    // ========================================================

    let drowsyScore = 0;

    // Strong downward head posture.
    if (
      features.pitchMean >
      18
    ) {

      drowsyScore += 0.60;

    } else if (
      features.pitchMean >
      14
    ) {

      drowsyScore += 0.40;

    } else if (
      features.pitchMean >
      12
    ) {

      drowsyScore += 0.20;
    }

    // Stable downward posture.
    if (
      features.pitchMean > 12 &&
      features.pitchStd < 3
    ) {

      drowsyScore += 0.15;
    }

    // Low gaze activity supports
    // drowsiness only when the head
    // is sufficiently downward.
    if (
      features.pitchMean > 12
    ) {

      if (
        features.gazeXStd < 20
      ) {
        drowsyScore += 0.10;
      }

      if (
        features.gazeYStd < 12
      ) {
        drowsyScore += 0.10;
      }
    }

    // Side-looking should not become DROWSY.
    if (
      Math.abs(features.yawMean) >
      15
    ) {

      drowsyScore *= 0.10;
    }

    drowsyScore =
      Math.min(
        drowsyScore,
        1
      );

    // ========================================================
    // DISTRACTION
    // ========================================================

    let distractionScore = 0;

    if (
      Math.abs(features.yawMean) >
      25
    ) {

      distractionScore += 0.55;

    } else if (
      Math.abs(features.yawMean) >
      18
    ) {

      distractionScore += 0.40;

    } else if (
      Math.abs(features.yawMean) >
      12
    ) {

      distractionScore += 0.25;
    }

    if (
      Math.abs(features.pitchMean) >
      15
    ) {

      distractionScore += 0.20;

    } else if (
      Math.abs(features.pitchMean) >
      10
    ) {

      distractionScore += 0.10;
    }

    if (
      features.gazeXStd >
      100
    ) {

      distractionScore += 0.20;

    } else if (
      features.gazeXStd >
      70
    ) {

      distractionScore += 0.10;
    }

    if (
      features.yawRange >
      30
    ) {

      distractionScore += 0.10;
    }

    if (
      features.pitchRange >
      20
    ) {

      distractionScore += 0.10;
    }

    distractionScore =
      Math.min(
        distractionScore,
        1
      );

    // ========================================================
    // CONCENTRATION
    // ========================================================

    let concentrationScore = 0;

    if (
      Math.abs(features.yawMean) <
      12
    ) {

      concentrationScore += 0.35;
    }

    if (
      Math.abs(features.pitchMean) <
      10
    ) {

      concentrationScore += 0.30;
    }

    if (
      features.yawStd <
      8
    ) {

      concentrationScore += 0.15;
    }

    if (
      features.pitchStd <
      6
    ) {

      concentrationScore += 0.10;
    }

    if (
      features.gazeXStd >= 10 &&
      features.gazeXStd <= 100
    ) {

      concentrationScore += 0.10;
    }

    concentrationScore =
      Math.min(
        concentrationScore,
        1
      );

    // ========================================================
    // MUTUAL EXCLUSION
    // ========================================================

    if (
      Math.abs(features.yawMean) >
      18
    ) {

      concentrationScore *= 0.25;
    }

    if (
      features.pitchMean >
      15
    ) {

      concentrationScore *= 0.25;
    }

    if (
      drowsyScore >= 0.70
    ) {

      concentrationScore *= 0.25;
      distractionScore *= 0.50;
    }

    if (
      distractionScore >= 0.50
    ) {

      concentrationScore *= 0.25;
    }

    const scores = {
      CONCENTRATED:
        concentrationScore,

      DISTRACTED:
        distractionScore,

      DROWSY:
        drowsyScore,
    };

    const state =
      Object.keys(scores).reduce(
        (best, current) =>
          scores[current] >
          scores[best]
            ? current
            : best
      );

    return {

      state,

      confidence:
        Number(
          scores[state].toFixed(4)
        ),

      scores: {
        CONCENTRATED:
          Number(
            concentrationScore.toFixed(4)
          ),

        DISTRACTED:
          Number(
            distractionScore.toFixed(4)
          ),

        DROWSY:
          Number(
            drowsyScore.toFixed(4)
          ),
      },

      frames:
        this.history.length,

      requiredFrames:
        this.windowSize,

      features,
    };
  }
}


// ============================================================
// DASHBOARD
// ============================================================

function Dashboard() {

  // ============================================================
  // CAMERA REFERENCES
  // ============================================================

  const driverVideoRef =
    useRef(null);

  const sceneVideoRef =
    useRef(null);

  const driverStreamRef =
    useRef(null);

  const sceneStreamRef =
    useRef(null);

  const driverCanvasRef =
    useRef(null);

  const sceneCanvasRef =
    useRef(null);


  // ============================================================
  // CONTINUOUS INFERENCE REFERENCES
  // ============================================================

  const monitoringRef =
    useRef(false);

  const inferenceRunningRef =
    useRef(false);

  const stateEngineRef =
    useRef(
      new PrototypeStateEngine(
        TEMPORAL_LENGTH
      )
    );


  // ============================================================
  // CAMERA STATE
  // ============================================================

  const [devices, setDevices] =
    useState([]);

  const [driverCamera, setDriverCamera] =
    useState("");

  const [sceneCamera, setSceneCamera] =
    useState("");

  const [camerasActive, setCamerasActive] =
    useState(false);

  const [driverReady, setDriverReady] =
    useState(false);

  const [sceneReady, setSceneReady] =
    useState(false);


  // ============================================================
  // BACKEND STATE
  // ============================================================

  const [backendOnline, setBackendOnline] =
    useState(false);

  const [checkingBackend, setCheckingBackend] =
    useState(true);


  // ============================================================
  // MONITORING STATE
  // ============================================================

  const [monitoring, setMonitoring] =
    useState(false);

  const [processing, setProcessing] =
    useState(false);

  const [framesProcessed, setFramesProcessed] =
    useState(0);

  const [inferenceFPS, setInferenceFPS] =
    useState(0);


  // ============================================================
  // MODEL OUTPUT
  // ============================================================

  const [prediction, setPrediction] =
    useState(null);

  const [heatmapImage, setHeatmapImage] =
    useState(null);

  const [sceneSnapshot, setSceneSnapshot] =
    useState(null);


  // ============================================================
  // DRIVER STATE
  // ============================================================

  const [driverState, setDriverState] =
    useState({
      state: "CALIBRATING",
      confidence: 0,
      scores: {
        CONCENTRATED: 0,
        DISTRACTED: 0,
        DROWSY: 0,
      },
      frames: 0,
      requiredFrames:
        TEMPORAL_LENGTH,
    });


  // ============================================================
  // ERROR
  // ============================================================

  const [error, setError] =
    useState("");

  const [lastInference, setLastInference] =
    useState(null);


  // ============================================================
  // INITIALIZE
  // ============================================================

  useEffect(() => {

    initializeDevices();

    return () => {

      monitoringRef.current =
        false;

      stopAllCameras();

    };

  }, []);


  // ============================================================
  // BACKEND MONITOR
  // ============================================================

  useEffect(() => {

    checkBackend();

    const interval =
      setInterval(
        () => {
          checkBackend();
        },
        10000
      );

    return () =>
      clearInterval(interval);

  }, []);


  // ============================================================
  // CAMERA ENUMERATION
  // ============================================================

  const initializeDevices =
    async () => {

      try {

        if (
          !navigator.mediaDevices?.getUserMedia
        ) {

          setError(
            "Camera access is not supported by this browser."
          );

          return;
        }

        // Request permission so that
        // Chrome can expose camera labels.
        try {

          const temporaryStream =
            await navigator.mediaDevices.getUserMedia(
              {
                video: true,
                audio: false,
              }
            );

          temporaryStream
            .getTracks()
            .forEach(
              (track) =>
                track.stop()
            );

        } catch (permissionError) {

          console.error(
            "Camera permission error:",
            permissionError
          );
        }

        const allDevices =
          await navigator.mediaDevices
            .enumerateDevices();

        const videoDevices =
          allDevices.filter(
            (device) =>
              device.kind ===
              "videoinput"
          );

        setDevices(
          videoDevices
        );

        if (
          videoDevices.length === 0
        ) {

          setError(
            "No camera detected. Connect at least one camera."
          );

          return;
        }

        setDriverCamera(
          videoDevices[0].deviceId
        );

        if (
          videoDevices.length > 1
        ) {

          setSceneCamera(
            videoDevices[1].deviceId
          );

        } else {

          setSceneCamera(
            videoDevices[0].deviceId
          );
        }

      } catch (err) {

        console.error(
          "Camera enumeration error:",
          err
        );

        setError(
          "Unable to access camera devices. Check browser permissions."
        );
      }
    };


  // ============================================================
  // BACKEND CHECK
  // ============================================================

  const checkBackend =
    async () => {

      setCheckingBackend(true);

      try {

        const result =
          await checkBackendStatus();

        setBackendOnline(
          result?.status ===
          "online"
        );

      } catch (err) {

        console.error(
          "Backend check failed:",
          err
        );

        setBackendOnline(false);

      } finally {

        setCheckingBackend(false);
      }
    };


  // ============================================================
  // START CAMERA
  // ============================================================

  const startCamera =
    async (
      deviceId,
      videoRef,
      type
    ) => {

      if (
        !deviceId ||
        !videoRef.current
      ) {

        return null;
      }

      try {

        const stream =
          await navigator.mediaDevices
            .getUserMedia({

              video: {

                deviceId: {
                  exact: deviceId,
                },

                width: {
                  ideal: 1280,
                },

                height: {
                  ideal: 720,
                },

                frameRate: {
                  ideal: 30,
                },
              },

              audio: false,
            });

        videoRef.current.srcObject =
          stream;

        await new Promise(
          (resolve) => {

            const video =
              videoRef.current;

            if (
              video.readyState >= 2
            ) {

              resolve();

            } else {

              video.onloadedmetadata =
                () => resolve();
            }
          }
        );

        await videoRef.current.play();

        if (
          type === "driver"
        ) {

          driverStreamRef.current =
            stream;

          setDriverReady(true);

        } else {

          sceneStreamRef.current =
            stream;

          setSceneReady(true);
        }

        return stream;

      } catch (err) {

        console.error(
          `${type} camera error:`,
          err
        );

        if (
          type === "driver"
        ) {

          setDriverReady(false);

        } else {

          setSceneReady(false);
        }

        throw err;
      }
    };


  // ============================================================
  // START BOTH CAMERAS
  // ============================================================

  const startAllCameras =
    async () => {

      setError("");

      if (
        !driverCamera ||
        !sceneCamera
      ) {

        setError(
          "Please select both cameras."
        );

        return;
      }

      if (
        devices.length > 1 &&
        driverCamera === sceneCamera
      ) {

        setError(
          "Driver Camera and Scene Camera must use different cameras."
        );

        return;
      }

      try {

        stopAllCameras();

        await startCamera(
          driverCamera,
          driverVideoRef,
          "driver"
        );

        await startCamera(
          sceneCamera,
          sceneVideoRef,
          "scene"
        );

        setCamerasActive(true);

      } catch (err) {

        console.error(
          "Failed to start cameras:",
          err
        );

        setCamerasActive(false);

        setError(
          "Unable to start both cameras. Check camera permissions and device selection."
        );
      }
    };


  // ============================================================
  // STOP CAMERAS
  // ============================================================

  const stopAllCameras =
    () => {

      // Stop monitoring first.
      monitoringRef.current =
        false;

      setMonitoring(false);
      setProcessing(false);

      if (
        driverStreamRef.current
      ) {

        driverStreamRef.current
          .getTracks()
          .forEach(
            (track) =>
              track.stop()
          );

        driverStreamRef.current =
          null;
      }

      if (
        sceneStreamRef.current
      ) {

        sceneStreamRef.current
          .getTracks()
          .forEach(
            (track) =>
              track.stop()
          );

        sceneStreamRef.current =
          null;
      }

      if (
        driverVideoRef.current
      ) {

        driverVideoRef.current
          .srcObject =
          null;
      }

      if (
        sceneVideoRef.current
      ) {

        sceneVideoRef.current
          .srcObject =
          null;
      }

      stateEngineRef.current.reset();

      setDriverState({
        state: "CALIBRATING",
        confidence: 0,
        scores: {
          CONCENTRATED: 0,
          DISTRACTED: 0,
          DROWSY: 0,
        },
        frames: 0,
        requiredFrames:
          TEMPORAL_LENGTH,
      });

      setDriverReady(false);
      setSceneReady(false);
      setCamerasActive(false);
    };


  // ============================================================
  // CAPTURE FRAME
  // ============================================================

  const captureFrame =
    (video, canvas) => {

      if (
        !video ||
        !canvas
      ) {

        throw new Error(
          "Video or canvas is unavailable."
        );
      }

      if (
        video.readyState < 2 ||
        video.videoWidth === 0 ||
        video.videoHeight === 0
      ) {

        throw new Error(
          "Camera frame is not ready yet."
        );
      }

      canvas.width =
        video.videoWidth;

      canvas.height =
        video.videoHeight;

      const context =
        canvas.getContext(
          "2d"
        );

      context.drawImage(
        video,
        0,
        0,
        canvas.width,
        canvas.height
      );

      return new Promise(
        (
          resolve,
          reject
        ) => {

          canvas.toBlob(
            (blob) => {

              if (!blob) {

                reject(
                  new Error(
                    "Could not capture camera frame."
                  )
                );

                return;
              }

              resolve(blob);
            },
            "image/jpeg",
            0.82
          );
        }
      );
    };


  // ============================================================
  // GAZE DIRECTION
  // ============================================================

  const getGazeDirection =
    (gazeLocation) => {

      if (
        !Array.isArray(
          gazeLocation
        ) ||
        gazeLocation.length < 2
      ) {

        return "UNKNOWN";
      }

      const x =
        Number(
          gazeLocation[0]
        );

      const y =
        Number(
          gazeLocation[1]
        );

      if (
        !Number.isFinite(x) ||
        !Number.isFinite(y)
      ) {

        return "UNKNOWN";
      }

      const sceneVideo =
        sceneVideoRef.current;

      const width =
        sceneVideo?.videoWidth ||
        942;

      const height =
        sceneVideo?.videoHeight ||
        489;

      const normalizedX =
        x / width;

      const normalizedY =
        y / height;

      if (
        normalizedX < 0.33
      ) {

        return "LEFT";
      }

      if (
        normalizedX > 0.67
      ) {

        return "RIGHT";
      }

      if (
        normalizedY < 0.33
      ) {

        return "UP";
      }

      if (
        normalizedY > 0.67
      ) {

        return "DOWN";
      }

      return "CENTER";
    };


  // ============================================================
  // CREATE SCENE SNAPSHOT
  // ============================================================

  const createSceneSnapshot =
    async () => {

      const blob =
        await captureFrame(
          sceneVideoRef.current,
          sceneCanvasRef.current
        );

      const url =
        URL.createObjectURL(
          blob
        );

      setSceneSnapshot(
        (previous) => {

          if (previous) {

            URL.revokeObjectURL(
              previous
            );
          }

          return url;
        }
      );

      return blob;
    };


  // ============================================================
  // PROCESS ONE CONTINUOUS FRAME
  // ============================================================

  const processOneFrame =
    async () => {

      if (
        !monitoringRef.current
      ) {

        return;
      }

      if (
        inferenceRunningRef.current
      ) {

        return;
      }

      if (
        !backendOnline
      ) {

        return;
      }

      if (
        !driverReady ||
        !sceneReady
      ) {

        return;
      }

      inferenceRunningRef.current =
        true;

      setProcessing(true);

      const startTime =
        performance.now();

      try {

        // ------------------------------------------------------
        // Capture synchronized camera frames
        // ------------------------------------------------------

        const driverFrame =
          await captureFrame(
            driverVideoRef.current,
            driverCanvasRef.current
          );

        const sceneFrame =
          await createSceneSnapshot();

        // ------------------------------------------------------
        // Send to actual L-DriX-Net backend
        // ------------------------------------------------------

        const result =
          await predictGaze(
            driverFrame,
            sceneFrame
          );

        console.log(
          "Continuous L-DriX-Net response:",
          result
        );

        // ------------------------------------------------------
        // Gaze
        // ------------------------------------------------------

        const gazeLocation =
          result?.gaze_location_2d ||
          result?.gaze?.gaze_location_2d ||
          null;

        const direction =
          getGazeDirection(
            gazeLocation
          );

        // ------------------------------------------------------
        // Face confidence
        // ------------------------------------------------------

        const faceConfidence =
          Number(
            result?.face_detection
              ?.confidence ?? 0
          );

        // ------------------------------------------------------
        // Heatmap
        // ------------------------------------------------------

        let heatmap =
          null;

        if (
          result
            ?.attention_heatmap
            ?.image_base64
        ) {

          heatmap =
            `data:image/png;base64,${result.attention_heatmap.image_base64}`;
        }

        if (heatmap) {

          setHeatmapImage(
            heatmap
          );
        }

        // ------------------------------------------------------
        // Prediction state
        // ------------------------------------------------------

        setPrediction({
          ...result,
          direction,
          gazeLocation,
          faceConfidence,
        });

        // ------------------------------------------------------
        // Prototype temporal state
        // ------------------------------------------------------

        const stateResult =
          stateEngineRef.current
            .update(result);

        setDriverState(
          stateResult
        );

        // ------------------------------------------------------
        // Statistics
        // ------------------------------------------------------

        const elapsed =
          performance.now() -
          startTime;

        const currentFPS =
          elapsed > 0
            ? 1000 / elapsed
            : 0;

        setInferenceFPS(
          Number(
            currentFPS.toFixed(1)
          )
        );

        setFramesProcessed(
          (previous) =>
            previous + 1
        );

        setLastInference(
          new Date()
            .toLocaleTimeString()
        );

        setError("");

      } catch (err) {

        console.error(
          "Continuous L-DriX-Net inference error:",
          err
        );

        setError(
          err?.message ||
            "Live inference failed. Check the backend terminal."
        );

      } finally {

        inferenceRunningRef.current =
          false;

        setProcessing(false);
      }
    };


  // ============================================================
  // CONTINUOUS MONITORING LOOP
  // ============================================================

  const monitoringLoop =
    async () => {

      while (
        monitoringRef.current
      ) {

        await processOneFrame();

        if (
          !monitoringRef.current
        ) {

          break;
        }

        await new Promise(
          (resolve) =>
            setTimeout(
              resolve,
              INFERENCE_INTERVAL
            )
        );
      }
    };


  // ============================================================
  // START MONITORING
  // ============================================================

  const startMonitoring =
    async () => {

      setError("");

      if (
        !backendOnline
      ) {

        setError(
          "Backend is offline. Start FastAPI on port 8000 first."
        );

        return;
      }

      if (
        !camerasActive ||
        !driverReady ||
        !sceneReady
      ) {

        setError(
          "Start both cameras before starting monitoring."
        );

        return;
      }

      if (
        monitoringRef.current
      ) {

        return;
      }

      // Reset temporal state
      stateEngineRef.current.reset();

      setDriverState({
        state: "CALIBRATING",
        confidence: 0,
        scores: {
          CONCENTRATED: 0,
          DISTRACTED: 0,
          DROWSY: 0,
        },
        frames: 0,
        requiredFrames:
          TEMPORAL_LENGTH,
      });

      setFramesProcessed(0);
      setInferenceFPS(0);

      monitoringRef.current =
        true;

      setMonitoring(true);

      // Start asynchronous loop.
      monitoringLoop();
    };


  // ============================================================
  // STOP MONITORING
  // ============================================================

  const stopMonitoring =
    () => {

      monitoringRef.current =
        false;

      setMonitoring(false);
      setProcessing(false);

      stateEngineRef.current.reset();

      setDriverState({
        state: "CALIBRATING",
        confidence: 0,
        scores: {
          CONCENTRATED: 0,
          DISTRACTED: 0,
          DROWSY: 0,
        },
        frames: 0,
        requiredFrames:
          TEMPORAL_LENGTH,
      });
    };


  // ============================================================
  // CLEAR RESULTS
  // ============================================================

  const clearResults =
    () => {

      setPrediction(null);
      setHeatmapImage(null);

      if (
        sceneSnapshot
      ) {

        URL.revokeObjectURL(
          sceneSnapshot
        );
      }

      setSceneSnapshot(null);

      setLastInference(null);
      setFramesProcessed(0);
      setInferenceFPS(0);

      stateEngineRef.current.reset();

      setDriverState({
        state: "CALIBRATING",
        confidence: 0,
        scores: {
          CONCENTRATED: 0,
          DISTRACTED: 0,
          DROWSY: 0,
        },
        frames: 0,
        requiredFrames:
          TEMPORAL_LENGTH,
      });

      setError("");
    };


  // ============================================================
  // CLEAN SNAPSHOT URL
  // ============================================================

  useEffect(() => {

    return () => {

      if (
        sceneSnapshot
      ) {

        URL.revokeObjectURL(
          sceneSnapshot
        );
      }
    };

  }, [sceneSnapshot]);


  // ============================================================
  // FORMAT NUMBER
  // ============================================================

  const formatNumber =
    (
      value,
      digits = 2
    ) => {

      const number =
        Number(value);

      if (
        !Number.isFinite(number)
      ) {

        return "—";
      }

      return number.toFixed(
        digits
      );
    };


  // ============================================================
  // TEMPORAL ATTENTION
  // ============================================================

  const formatTemporalAttention =
    (value) => {

      if (
        value === null ||
        value === undefined
      ) {

        return "—";
      }

      if (
        Array.isArray(value)
      ) {

        if (
          value.length === 0
        ) {

          return "—";
        }

        return value
          .map(
            (item) =>
              formatNumber(
                item,
                3
              )
          )
          .join(", ");
      }

      if (
        typeof value ===
        "number"
      ) {

        return formatNumber(
          value,
          3
        );
      }

      if (
        typeof value ===
        "string"
      ) {

        const numericValue =
          Number(value);

        return Number.isFinite(
          numericValue
        )
          ? formatNumber(
              numericValue,
              3
            )
          : value;
      }

      if (
        typeof value ===
        "object"
      ) {

        try {

          return JSON.stringify(
            value
          );

        } catch {

          return "Available";
        }
      }

      return String(value);
    };


  // ============================================================
  // STATE COLOR
  // ============================================================

  const getStateColor =
    (state) => {

      if (
        state ===
        "CONCENTRATED"
      ) {

        return "#75d69c";
      }

      if (
        state ===
        "DISTRACTED"
      ) {

        return "#f0c674";
      }

      if (
        state ===
        "DROWSY"
      ) {

        return "#e98282";
      }

      return "#858e9d";
    };


  // ============================================================
  // STATE LABEL
  // ============================================================

  const getStateDescription =
    (state) => {

      if (
        state ===
        "CONCENTRATED"
      ) {

        return "Driver attention appears directed toward the road.";
      }

      if (
        state ===
        "DISTRACTED"
      ) {

        return "Temporal head/gaze behavior indicates diverted attention.";
      }

      if (
        state ===
        "DROWSY"
      ) {

        return "Prototype low-alertness pattern detected.";
      }

      return "Collecting temporal driver behavior...";
    };


  // ============================================================
  // RENDER
  // ============================================================

  return (

    <div
      style={{
        minHeight: "100%",
        padding: "28px",
        background: "#0f1115",
        color: "#f3f4f6",
        fontFamily:
          "Inter, system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
      }}
    >

      {/* ======================================================
          HEADER
      ====================================================== */}

      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          gap: "20px",
          marginBottom: "28px",
        }}
      >

        <div>

          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              fontSize: "12px",
              fontWeight: 700,
              letterSpacing: "1.5px",
              color: "#8b95a7",
              marginBottom: "8px",
            }}
          >

            <Brain size={15} />

            L-DRIX-NET

          </div>

          <h1
            style={{
              margin: 0,
              fontSize: "32px",
              fontWeight: 750,
              letterSpacing: "-0.8px",
            }}
          >

            Driver Attention Dashboard

          </h1>

          <p
            style={{
              margin:
                "8px 0 0",
              color: "#8f98a8",
              fontSize: "14px",
            }}
          >

            Continuous dual-camera gaze estimation
            and temporal driver-state monitoring.

          </p>

        </div>


        {/* Backend status */}

        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "10px",
            padding:
              "10px 14px",
            borderRadius: "10px",
            border:
              "1px solid #292e38",
            background: "#15181e",
            fontSize: "13px",
          }}
        >

          {checkingBackend ? (

            <>

              <RefreshCw
                size={15}
              />

              Checking backend...

            </>

          ) : backendOnline ? (

            <>

              <Wifi size={15} />

              <span
                style={{
                  color:
                    "#9fe3bd",
                }}
              >
                Backend Online
              </span>

            </>

          ) : (

            <>

              <WifiOff
                size={15}
              />

              <span
                style={{
                  color:
                    "#f2a5a5",
                }}
              >
                Backend Offline
              </span>

            </>

          )}

        </div>

      </div>


      {/* ======================================================
          ERROR
      ====================================================== */}

      {error && (

        <div
          style={{
            display: "flex",
            alignItems:
              "flex-start",
            gap: "10px",
            padding:
              "14px 16px",
            marginBottom: "22px",
            borderRadius: "10px",
            border:
              "1px solid #5b3030",
            background: "#241719",
            color: "#f3b1b1",
            fontSize: "13px",
          }}
        >

          <AlertCircle
            size={18}
          />

          <div>

            <strong>
              System message
            </strong>

            <div
              style={{
                marginTop: "3px",
              }}
            >
              {error}
            </div>

          </div>

        </div>

      )}


      {/* ======================================================
          LIVE DRIVER STATE
      ====================================================== */}

      <section
        style={{
          marginBottom: "24px",
          padding: "22px",
          borderRadius: "14px",
          border:
            "1px solid #272c35",
          background: "#15181e",
        }}
      >

        <div
          style={{
            display: "flex",
            justifyContent:
              "space-between",
            alignItems: "center",
            gap: "20px",
            flexWrap: "wrap",
          }}
        >

          <div>

            <div
              style={{
                display: "flex",
                alignItems:
                  "center",
                gap: "9px",
              }}
            >

              <Activity
                size={20}
              />

              <h2
                style={{
                  margin: 0,
                  fontSize: "18px",
                }}
              >
                Driver State
              </h2>

            </div>

            <p
              style={{
                margin:
                  "6px 0 0",
                color:
                  "#858e9d",
                fontSize: "13px",
              }}
            >
              Prototype temporal state engine using a
              {` ${TEMPORAL_LENGTH}`}-frame window.
            </p>

          </div>


          <div
            style={{
              display: "flex",
              alignItems:
                "center",
              gap: "10px",
            }}
          >

            <span
              style={{
                width: "10px",
                height: "10px",
                borderRadius:
                  "50%",
                background:
                  monitoring
                    ? "#75d69c"
                    : "#555c68",
              }}
            />

            <span
              style={{
                fontSize: "12px",
                color:
                  monitoring
                    ? "#9fe3bd"
                    : "#858e9d",
                fontWeight: 700,
              }}
            >
              {monitoring
                ? "MONITORING"
                : "STANDBY"}
            </span>

          </div>

        </div>


        {/* State display */}

        <div
          style={{
            display: "grid",
            gridTemplateColumns:
              "minmax(220px, 1.4fr) repeat(3, minmax(130px, 1fr))",
            gap: "12px",
            marginTop: "20px",
          }}
        >

          <div
            style={{
              padding: "20px",
              borderRadius: "12px",
              border:
                `1px solid ${getStateColor(
                  driverState.state
                )}55`,
              background:
                "#101319",
            }}
          >

            <div
              style={{
                fontSize: "11px",
                color:
                  "#7f8897",
                textTransform:
                  "uppercase",
                letterSpacing:
                  "0.7px",
              }}
            >
              Current State
            </div>

            <div
              style={{
                marginTop:
                  "8px",
                fontSize: "25px",
                fontWeight: 800,
                color:
                  getStateColor(
                    driverState.state
                  ),
              }}
            >
              {driverState.state}
            </div>

            <div
              style={{
                marginTop:
                  "7px",
                fontSize: "12px",
                color:
                  "#858e9d",
                lineHeight: 1.5,
              }}
            >
              {getStateDescription(
                driverState.state
              )}
            </div>

          </div>


          <StateScoreCard
            title="CONCENTRATED"
            value={
              driverState
                .scores
                ?.CONCENTRATED ??
              0
            }
          />

          <StateScoreCard
            title="DISTRACTED"
            value={
              driverState
                .scores
                ?.DISTRACTED ??
              0
            }
          />

          <StateScoreCard
            title="DROWSY"
            value={
              driverState
                .scores
                ?.DROWSY ??
              0
            }
          />

        </div>


        {/* Temporal progress */}

        <div
          style={{
            marginTop:
              "16px",
          }}
        >

          <div
            style={{
              display: "flex",
              justifyContent:
                "space-between",
              fontSize: "11px",
              color:
                "#858e9d",
              marginBottom:
                "7px",
            }}
          >

            <span>
              Temporal Window
            </span>

            <span>
              {driverState.frames}
              {" / "}
              {driverState.requiredFrames}
              {" frames"}
            </span>

          </div>

          <div
            style={{
              width: "100%",
              height: "7px",
              borderRadius:
                "10px",
              background:
                "#252a32",
              overflow:
                "hidden",
            }}
          >

            <div
              style={{
                width:
                  `${Math.min(
                    100,
                    (
                      driverState.frames /
                      driverState.requiredFrames
                    ) * 100
                  )}%`,
                height: "100%",
                background:
                  getStateColor(
                    driverState.state
                  ),
              }}
            />

          </div>

        </div>

      </section>


      {/* ======================================================
          CAMERA SECTION
      ====================================================== */}

      <section
        style={{
          marginBottom:
            "24px",
          padding: "20px",
          borderRadius:
            "14px",
          border:
            "1px solid #272c35",
          background:
            "#15181e",
        }}
      >

        <div
          style={{
            display: "flex",
            justifyContent:
              "space-between",
            alignItems:
              "center",
            marginBottom:
              "18px",
            gap: "15px",
          }}
        >

          <div>

            <div
              style={{
                display: "flex",
                alignItems:
                  "center",
                gap: "9px",
              }}
            >

              <Camera
                size={19}
              />

              <h2
                style={{
                  margin: 0,
                  fontSize: "18px",
                }}
              >
                Live Camera Inputs
              </h2>

            </div>

            <p
              style={{
                margin:
                  "6px 0 0 28px",
                color:
                  "#858e9d",
                fontSize: "13px",
              }}
            >
              Camera 1 observes the driver.
              Camera 2 observes the road scene.
            </p>

          </div>


          <div
            style={{
              display: "flex",
              alignItems:
                "center",
              gap: "8px",
              color:
                camerasActive
                  ? "#9fe3bd"
                  : "#858e9d",
              fontSize: "13px",
            }}
          >

            <span
              style={{
                width: "8px",
                height: "8px",
                borderRadius:
                  "50%",
                background:
                  camerasActive
                    ? "#74d69c"
                    : "#555c68",
              }}
            />

            {camerasActive
              ? "CAMERAS ACTIVE"
              : "CAMERAS STOPPED"}

          </div>

        </div>


        {/* Camera selectors */}

        <div
          style={{
            display: "grid",
            gridTemplateColumns:
              "repeat(2, minmax(0, 1fr))",
            gap: "16px",
            marginBottom:
              "18px",
          }}
        >

          <CameraSelector
            label="Driver Camera"
            value={driverCamera}
            devices={devices}
            onChange={
              setDriverCamera
            }
            disabled={
              camerasActive
            }
          />

          <CameraSelector
            label="Scene Camera"
            value={sceneCamera}
            devices={devices}
            onChange={
              setSceneCamera
            }
            disabled={
              camerasActive
            }
          />

        </div>


        {/* Live videos */}

        <div
          style={{
            display: "grid",
            gridTemplateColumns:
              "repeat(2, minmax(0, 1fr))",
            gap: "16px",
          }}
        >

          <VideoPanel
            title="Driver View"
            subtitle="Face / Eye Input"
            videoRef={
              driverVideoRef
            }
            active={
              driverReady
            }
          />

          <VideoPanel
            title="Road Scene"
            subtitle="Scene Input"
            videoRef={
              sceneVideoRef
            }
            active={
              sceneReady
            }
          />

        </div>


        {/* Controls */}

        <div
          style={{
            display: "flex",
            justifyContent:
              "center",
            alignItems:
              "center",
            gap: "10px",
            marginTop:
              "18px",
            flexWrap:
              "wrap",
          }}
        >

          {!camerasActive ? (

            <button
              type="button"
              onClick={
                startAllCameras
              }
              style={
                primaryButtonStyle
              }
            >

              <Play size={16} />

              Start Dual Cameras

            </button>

          ) : (

            <button
              type="button"
              onClick={
                stopAllCameras
              }
              style={
                secondaryButtonStyle
              }
            >

              <Square size={15} />

              Stop Cameras

            </button>

          )}


          {camerasActive &&
            !monitoring ? (

            <button
              type="button"
              onClick={
                startMonitoring
              }
              disabled={
                !backendOnline
              }
              style={{
                ...primaryButtonStyle,
                opacity:
                  !backendOnline
                    ? 0.45
                    : 1,
                cursor:
                  !backendOnline
                    ? "not-allowed"
                    : "pointer",
              }}
            >

              <Brain
                size={16}
              />

              Start Live Monitoring

            </button>

          ) : camerasActive ? (

            <button
              type="button"
              onClick={
                stopMonitoring
              }
              style={
                secondaryButtonStyle
              }
            >

              <Square
                size={15}
              />

              Stop Monitoring

            </button>

          ) : null}


          <button
            type="button"
            onClick={
              clearResults
            }
            style={
              secondaryButtonStyle
            }
          >

            <RefreshCw
              size={15}
            />

            Clear Results

          </button>

        </div>


        {/* Live processing status */}

        {monitoring && (

          <div
            style={{
              display: "flex",
              justifyContent:
                "center",
              alignItems:
                "center",
              gap: "14px",
              marginTop:
                "14px",
              color:
                "#858e9d",
              fontSize: "12px",
            }}
          >

            <span
              style={{
                display:
                  "inline-flex",
                alignItems:
                  "center",
                gap: "6px",
              }}
            >

              <span
                style={{
                  width: "7px",
                  height: "7px",
                  borderRadius:
                    "50%",
                  background:
                    processing
                      ? "#75d69c"
                      : "#555c68",
                }}
              />

              {processing
                ? "L-DriX-Net processing..."
                : "Waiting for next frame"}

            </span>

            <span>
              Frames:
              {" "}
              {framesProcessed}
            </span>

            <span>
              Inference:
              {" "}
              {inferenceFPS}
              {" FPS"}
            </span>

          </div>

        )}

      </section>


      {/* Hidden canvases */}

      <canvas
        ref={
          driverCanvasRef
        }
        style={{
          display: "none",
        }}
      />

      <canvas
        ref={
          sceneCanvasRef
        }
        style={{
          display: "none",
        }}
      />


      {/* ======================================================
          LIVE GAZE OUTPUT
      ====================================================== */}

      <section
        style={{
          marginBottom:
            "24px",
          padding: "20px",
          borderRadius:
            "14px",
          border:
            "1px solid #272c35",
          background:
            "#15181e",
        }}
      >

        <div
          style={{
            display: "flex",
            justifyContent:
              "space-between",
            alignItems:
              "center",
            marginBottom:
              "20px",
          }}
        >

          <div
            style={{
              display: "flex",
              alignItems:
                "center",
              gap: "9px",
            }}
          >

            <Target
              size={19}
            />

            <div>

              <h2
                style={{
                  margin: 0,
                  fontSize: "18px",
                }}
              >
                Live Gaze Prediction
              </h2>

              <p
                style={{
                  margin:
                    "5px 0 0",
                  color:
                    "#858e9d",
                  fontSize: "13px",
                }}
              >
                Continuous L-DriX-Net inference output.
              </p>

            </div>

          </div>


          {lastInference && (

            <span
              style={{
                color:
                  "#858e9d",
                fontSize: "12px",
              }}
            >
              Last inference:
              {" "}
              {lastInference}
            </span>

          )}

        </div>


        <div
          style={{
            display: "grid",
            gridTemplateColumns:
              "repeat(6, minmax(0, 1fr))",
            gap: "12px",
          }}
        >

          <ResultCard
            icon={
              <Eye size={18} />
            }
            label="Direction"
            value={
              prediction
                ?.direction ||
              "—"
            }
          />

          <ResultCard
            icon={
              <Target size={18} />
            }
            label="Gaze X"
            value={
              prediction
                ?.gazeLocation
                ? formatNumber(
                    prediction
                      .gazeLocation[0],
                    1
                  )
                : "—"
            }
          />

          <ResultCard
            icon={
              <Target size={18} />
            }
            label="Gaze Y"
            value={
              prediction
                ?.gazeLocation
                ? formatNumber(
                    prediction
                      .gazeLocation[1],
                    1
                  )
                : "—"
            }
          />

          <ResultCard
            icon={
              <Gauge size={18} />
            }
            label="Face Confidence"
            value={
              prediction
                ? `${formatNumber(
                    Number(
                      prediction.faceConfidence
                    ) * 100,
                    1
                  )}%`
                : "—"
            }
          />

          <ResultCard
            label="Yaw"
            value={
              prediction
                ?.head_pose
                ?.yaw !==
              undefined
                ? `${formatNumber(
                    prediction
                      .head_pose
                      .yaw,
                    1
                  )}°`
                : "—"
            }
          />

          <ResultCard
            label="Pitch"
            value={
              prediction
                ?.head_pose
                ?.pitch !==
              undefined
                ? `${formatNumber(
                    prediction
                      .head_pose
                      .pitch,
                    1
                  )}°`
                : "—"
            }
          />

        </div>

      </section>


      {/* ======================================================
          VISUALIZATION
      ====================================================== */}

      <section
        style={{
          display: "grid",
          gridTemplateColumns:
            "repeat(2, minmax(0, 1fr))",
          gap: "20px",
          marginBottom:
            "24px",
        }}
      >

        <VisualizationCard
          title="Live Scene Frame"
          subtitle="Latest scene frame submitted to L-DriX-Net"
        >

          {sceneSnapshot ? (

            <img
              src={
                sceneSnapshot
              }
              alt="Latest captured road scene"
              style={{
                width: "100%",
                aspectRatio:
                  "16 / 9",
                objectFit:
                  "contain",
                background:
                  "#090b0f",
                borderRadius:
                  "10px",
              }}
            />

          ) : (

            <EmptyVisualization
              icon={
                <Monitor
                  size={30}
                />
              }
              text="Start live monitoring to capture the road scene."
            />

          )}

        </VisualizationCard>


        <VisualizationCard
          title="L-DriX-Net Attention Heatmap"
          subtitle="Latest predicted spatial attention"
        >

          {heatmapImage ? (

            <img
              src={
                heatmapImage
              }
              alt="L-DriX-Net attention heatmap"
              style={{
                width: "100%",
                aspectRatio:
                  "1 / 1",
                objectFit:
                  "contain",
                background:
                  "#090b0f",
                borderRadius:
                  "10px",
              }}
            />

          ) : (

            <EmptyVisualization
              icon={
                <Target
                  size={30}
                />
              }
              text="Start live monitoring to generate the attention heatmap."
            />

          )}

        </VisualizationCard>

      </section>


      {/* ======================================================
          MODEL OUTPUT
      ====================================================== */}

      <section
        style={{
          marginBottom:
            "24px",
          padding: "20px",
          borderRadius:
            "14px",
          border:
            "1px solid #272c35",
          background:
            "#15181e",
        }}
      >

        <div
          style={{
            display: "flex",
            alignItems:
              "center",
            gap: "9px",
            marginBottom:
              "18px",
          }}
        >

          <Activity
            size={19}
          />

          <div>

            <h2
              style={{
                margin: 0,
                fontSize: "18px",
              }}
            >
              Model Output
            </h2>

            <p
              style={{
                margin:
                  "5px 0 0",
                color:
                  "#858e9d",
                fontSize: "13px",
              }}
            >
              L-DriX-Net prediction tensor and temporal information.
            </p>

          </div>

        </div>


        <div
          style={{
            display: "grid",
            gridTemplateColumns:
              "repeat(3, minmax(0, 1fr))",
            gap: "12px",
          }}
        >

          <ResultCard
            label="Prediction Dimension"
            value={
              prediction
                ?.prediction
                ?.length
                ? `${prediction.prediction.length} values`
                : "24 values"
            }
          />

          <ResultCard
            label="Heatmap Size"
            value={
              prediction
                ?.attention_heatmap
                ?.shape
                ? prediction
                    .attention_heatmap
                    .shape
                    .slice(-2)
                    .join(" × ")
                : "224 × 224"
            }
          />

          <ResultCard
            label="Temporal Attention"
            value={
              formatTemporalAttention(
                prediction
                  ?.temporal_attention
              )
            }
          />

        </div>

      </section>


      {/* ======================================================
          MODEL BENCHMARK
      ====================================================== */}

      <section
        style={{
          marginBottom:
            "24px",
          padding: "20px",
          borderRadius:
            "14px",
          border:
            "1px solid #272c35",
          background:
            "#15181e",
        }}
      >

        <div
          style={{
            display: "flex",
            alignItems:
              "center",
            gap: "9px",
            marginBottom:
              "6px",
          }}
        >

          <Gauge
            size={19}
          />

          <h2
            style={{
              margin: 0,
              fontSize: "18px",
            }}
          >
            Model Evaluation
          </h2>

        </div>


        <p
          style={{
            margin:
              "0 0 18px",
            color:
              "#858e9d",
            fontSize: "13px",
          }}
        >
          Benchmark values from the trained L-DriX-Net evaluation.
          These are not per-frame live metrics because live frames
          do not contain ground-truth labels.
        </p>


        <div
          style={{
            display: "grid",
            gridTemplateColumns:
              "repeat(4, minmax(0, 1fr))",
            gap: "12px",
          }}
        >

          <ResultCard
            label="MAE"
            value="9.9573"
          />

          <ResultCard
            label="RMSE"
            value="45.1664"
          />

          <ResultCard
            label="KL Divergence"
            value="0.0012"
          />

          <ResultCard
            label="NCC"
            value="0.2524"
          />

        </div>

      </section>


      {/* ======================================================
          PIPELINE
      ====================================================== */}

      <section
        style={{
          padding: "20px",
          borderRadius:
            "14px",
          border:
            "1px solid #272c35",
          background:
            "#15181e",
        }}
      >

        <div
          style={{
            display: "flex",
            alignItems:
              "center",
            gap: "9px",
            marginBottom:
              "20px",
          }}
        >

          <Brain
            size={19}
          />

          <div>

            <h2
              style={{
                margin: 0,
                fontSize: "18px",
              }}
            >
              Live L-DriX-Net Processing Pipeline
            </h2>

            <p
              style={{
                margin:
                  "5px 0 0",
                color:
                  "#858e9d",
                fontSize: "13px",
              }}
            >
              Continuous dual-camera inference flow
            </p>

          </div>

        </div>


        <div
          style={{
            display: "grid",
            gridTemplateColumns:
              "repeat(5, minmax(0, 1fr))",
            gap: "10px",
          }}
        >

          <PipelineStep
            number="01"
            title="Driver Camera"
            subtitle="Face & eye input"
            active={
              driverReady
            }
          />

          <PipelineStep
            number="02"
            title="Scene Camera"
            subtitle="Road scene input"
            active={
              sceneReady
            }
          />

          <PipelineStep
            number="03"
            title="L-DriX-Net"
            subtitle="Multimodal fusion"
            active={
              processing
            }
          />

          <PipelineStep
            number="04"
            title="Temporal Window"
            subtitle="16-frame behavior"
            active={
              driverState.frames >=
              TEMPORAL_LENGTH
            }
          />

          <PipelineStep
            number="05"
            title="Driver State"
            subtitle="Attention classification"
            active={
              driverState.state !==
              "CALIBRATING"
            }
          />

        </div>

      </section>


      {/* ======================================================
          FOOTER
      ====================================================== */}

      <div
        style={{
          display: "flex",
          justifyContent:
            "space-between",
          alignItems:
            "center",
          marginTop:
            "20px",
          padding:
            "12px 4px",
          color:
            "#737d8d",
          fontSize: "12px",
        }}
      >

        <div
          style={{
            display: "flex",
            alignItems:
              "center",
            gap: "7px",
          }}
        >

          <CheckCircle2
            size={15}
          />

          <span>
            L-DriX-Net prototype ready
          </span>

        </div>

        <span>
          Continuous Dual Camera Mode
        </span>

      </div>

    </div>
  );
}


// ============================================================
// STATE SCORE CARD
// ============================================================

function StateScoreCard({
  title,
  value,
}) {

  const percentage =
    Math.round(
      Number(value || 0) *
      100
    );

  return (

    <div
      style={{
        padding: "18px",
        borderRadius:
          "12px",
        border:
          "1px solid #292f39",
        background:
          "#101319",
      }}
    >

      <div
        style={{
          fontSize: "11px",
          color:
            "#7f8897",
          textTransform:
            "uppercase",
          letterSpacing:
            "0.7px",
        }}
      >
        {title}
      </div>

      <div
        style={{
          marginTop:
            "8px",
          fontSize:
            "24px",
          fontWeight: 750,
        }}
      >
        {percentage}%
      </div>

      <div
        style={{
          marginTop:
            "10px",
          width: "100%",
          height: "5px",
          borderRadius:
            "10px",
          background:
            "#252a32",
          overflow:
            "hidden",
        }}
      >

        <div
          style={{
            width:
              `${Math.min(
                100,
                percentage
              )}%`,
            height: "100%",
            background:
              "#858e9d",
          }}
        />

      </div>

    </div>
  );
}


// ============================================================
// CAMERA SELECTOR
// ============================================================

function CameraSelector({
  label,
  value,
  devices,
  onChange,
  disabled,
}) {

  return (

    <label
      style={{
        display: "flex",
        flexDirection:
          "column",
        gap: "7px",
      }}
    >

      <span
        style={{
          fontSize: "12px",
          color:
            "#929baa",
          fontWeight: 600,
        }}
      >
        {label}
      </span>

      <select
        value={value}
        onChange={(event) =>
          onChange(
            event.target.value
          )
        }
        disabled={disabled}
        style={{
          width: "100%",
          padding:
            "10px 12px",
          borderRadius:
            "8px",
          border:
            "1px solid #303641",
          background:
            "#0e1116",
          color:
            "#e7eaf0",
          outline: "none",
          fontSize: "13px",
        }}
      >

        {devices.length === 0 ? (

          <option value="">
            No camera detected
          </option>

        ) : (

          devices.map(
            (
              device,
              index
            ) => (

              <option
                key={
                  device.deviceId
                }
                value={
                  device.deviceId
                }
              >

                {device.label ||
                  `Camera ${index + 1}`}

              </option>

            )
          )

        )}

      </select>

    </label>
  );
}


// ============================================================
// VIDEO PANEL
// ============================================================

function VideoPanel({
  title,
  subtitle,
  videoRef,
  active,
}) {

  return (

    <div
      style={{
        position:
          "relative",
        overflow:
          "hidden",
        borderRadius:
          "10px",
        border:
          "1px solid #292f39",
        background:
          "#090b0f",
      }}
    >

      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        style={{
          display: "block",
          width: "100%",
          aspectRatio:
            "16 / 9",
          objectFit:
            "cover",
          background:
            "#090b0f",
        }}
      />


      {!active && (

        <div
          style={{
            position:
              "absolute",
            inset: 0,
            display: "flex",
            flexDirection:
              "column",
            alignItems:
              "center",
            justifyContent:
              "center",
            gap: "8px",
            background:
              "#090b0f",
            color:
              "#626b79",
          }}
        >

          <Camera
            size={30}
          />

          <span
            style={{
              fontSize:
                "13px",
            }}
          >
            Camera inactive
          </span>

        </div>

      )}


      <div
        style={{
          position:
            "absolute",
          left: "10px",
          top: "10px",
          padding:
            "7px 9px",
          borderRadius:
            "7px",
          background:
            "rgba(0,0,0,0.65)",
          backdropFilter:
            "blur(5px)",
        }}
      >

        <div
          style={{
            fontSize:
              "12px",
            fontWeight:
              700,
          }}
        >
          {title}
        </div>

        <div
          style={{
            marginTop:
              "2px",
            fontSize:
              "10px",
            color:
              "#a0a8b5",
          }}
        >
          {subtitle}
        </div>

      </div>


      <div
        style={{
          position:
            "absolute",
          right: "10px",
          top: "10px",
          display: "flex",
          alignItems:
            "center",
          gap: "6px",
          padding:
            "6px 8px",
          borderRadius:
            "7px",
          background:
            "rgba(0,0,0,0.65)",
          fontSize:
            "10px",
        }}
      >

        <span
          style={{
            width: "6px",
            height: "6px",
            borderRadius:
              "50%",
            background:
              active
                ? "#74d69c"
                : "#555c68",
          }}
        />

        {active
          ? "LIVE"
          : "OFFLINE"}

      </div>

    </div>
  );
}


// ============================================================
// RESULT CARD
// ============================================================

function ResultCard({
  icon,
  label,
  value,
}) {

  return (

    <div
      style={{
        padding:
          "15px",
        borderRadius:
          "10px",
        border:
          "1px solid #292f39",
        background:
          "#101319",
      }}
    >

      {icon && (

        <div
          style={{
            marginBottom:
              "10px",
            color:
              "#929baa",
          }}
        >
          {icon}
        </div>

      )}

      <div
        style={{
          fontSize:
            "11px",
          color:
            "#7f8897",
          textTransform:
            "uppercase",
          letterSpacing:
            "0.7px",
        }}
      >
        {label}
      </div>

      <div
        style={{
          marginTop:
            "6px",
          fontSize:
            "20px",
          fontWeight:
            700,
          color:
            "#eef1f5",
          wordBreak:
            "break-word",
        }}
      >
        {value}
      </div>

    </div>
  );
}


// ============================================================
// VISUALIZATION CARD
// ============================================================

function VisualizationCard({
  title,
  subtitle,
  children,
}) {

  return (

    <div
      style={{
        padding:
          "18px",
        borderRadius:
          "14px",
        border:
          "1px solid #272c35",
        background:
          "#15181e",
      }}
    >

      <div
        style={{
          marginBottom:
            "14px",
        }}
      >

        <h3
          style={{
            margin: 0,
            fontSize:
              "16px",
          }}
        >
          {title}
        </h3>

        <p
          style={{
            margin:
              "4px 0 0",
            color:
              "#858e9d",
            fontSize:
              "12px",
          }}
        >
          {subtitle}
        </p>

      </div>

      {children}

    </div>
  );
}


// ============================================================
// EMPTY VISUALIZATION
// ============================================================

function EmptyVisualization({
  icon,
  text,
}) {

  return (

    <div
      style={{
        display: "flex",
        flexDirection:
          "column",
        alignItems:
          "center",
        justifyContent:
          "center",
        minHeight:
          "300px",
        gap: "12px",
        borderRadius:
          "10px",
        background:
          "#090b0f",
        color:
          "#626b79",
      }}
    >

      {icon}

      <span
        style={{
          fontSize:
            "12px",
          textAlign:
            "center",
          maxWidth:
            "260px",
        }}
      >
        {text}
      </span>

    </div>
  );
}


// ============================================================
// PIPELINE STEP
// ============================================================

function PipelineStep({
  number,
  title,
  subtitle,
  active,
}) {

  return (

    <div
      style={{
        padding:
          "15px",
        borderRadius:
          "10px",
        border:
          "1px solid #292f39",
        background:
          "#101319",
      }}
    >

      <div
        style={{
          display:
            "inline-flex",
          alignItems:
            "center",
          justifyContent:
            "center",
          width:
            "28px",
          height:
            "28px",
          borderRadius:
            "7px",
          background:
            active
              ? "#2b3a33"
              : "#20252e",
          color:
            active
              ? "#9fe3bd"
              : "#c4cad3",
          fontSize:
            "11px",
          fontWeight:
            700,
          marginBottom:
            "12px",
        }}
      >
        {number}
      </div>

      <div
        style={{
          fontSize:
            "13px",
          fontWeight:
            700,
        }}
      >
        {title}
      </div>

      <div
        style={{
          marginTop:
            "4px",
          fontSize:
            "11px",
          color:
            "#777f8e",
        }}
      >
        {subtitle}
      </div>

    </div>
  );
}


// ============================================================
// BUTTON STYLES
// ============================================================

const primaryButtonStyle = {

  display:
    "inline-flex",

  alignItems:
    "center",

  justifyContent:
    "center",

  gap: "8px",

  padding:
    "11px 16px",

  borderRadius:
    "8px",

  border:
    "1px solid #3a414d",

  background:
    "#e8ebef",

  color:
    "#111318",

  fontSize:
    "13px",

  fontWeight:
    700,

  cursor:
    "pointer",
};


const secondaryButtonStyle = {

  display:
    "inline-flex",

  alignItems:
    "center",

  justifyContent:
    "center",

  gap: "8px",

  padding:
    "11px 16px",

  borderRadius:
    "8px",

  border:
    "1px solid #343a45",

  background:
    "#1b1f26",

  color:
    "#d9dde4",

  fontSize:
    "13px",

  fontWeight:
    600,

  cursor:
    "pointer",
};


export default Dashboard;