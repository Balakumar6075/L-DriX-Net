import {
  useEffect,
  useRef,
  useState,
} from "react";

import {
  Camera,
  CameraOff,
  RefreshCw,
} from "lucide-react";


function CameraFeed({
  cameraType,
  isActive,
  onStreamReady,
  onFrameReady,
}) {

  const videoRef = useRef(null);

  const streamRef = useRef(null);

  const [devices, setDevices] = useState([]);

  const [selectedDevice, setSelectedDevice] =
    useState("");

  const [cameraStatus, setCameraStatus] =
    useState("idle");

  const [error, setError] =
    useState(null);


  // ==========================================================
  // CAMERA LABEL
  // ==========================================================

  const title =
    cameraType === "driver"
      ? "Driver Camera"
      : "Scene Camera";


  const description =
    cameraType === "driver"
      ? "Face, eyes & head pose"
      : "Road & scene environment";


  // ==========================================================
  // GET CAMERAS
  // ==========================================================

  const loadDevices = async () => {

    try {

      const mediaDevices =
        await navigator.mediaDevices.enumerateDevices();

      const videoDevices =
        mediaDevices.filter(
          (device) =>
            device.kind === "videoinput"
        );

      setDevices(videoDevices);

      // If no camera is currently selected,
      // choose a sensible default.

      if (!selectedDevice && videoDevices.length > 0) {

        if (
          cameraType === "scene" &&
          videoDevices.length > 1
        ) {

          setSelectedDevice(
            videoDevices[1].deviceId
          );

        } else {

          setSelectedDevice(
            videoDevices[0].deviceId
          );

        }

      }

    } catch (err) {

      console.error(
        "Camera enumeration error:",
        err
      );

      setError(
        "Unable to access camera devices."
      );

    }

  };


  // ==========================================================
  // INITIAL DEVICE ENUMERATION
  // ==========================================================

  useEffect(() => {

    if (
      !navigator.mediaDevices ||
      !navigator.mediaDevices.enumerateDevices
    ) {

      setError(
        "Camera access is not supported by this browser."
      );

      return;

    }

    loadDevices();

  }, []);


  // ==========================================================
  // START CAMERA
  // ==========================================================

  const startCamera = async () => {

    try {

      setError(null);

      setCameraStatus("starting");

      // Stop previous stream first.

      if (streamRef.current) {

        streamRef.current
          .getTracks()
          .forEach(
            (track) =>
              track.stop()
          );

        streamRef.current = null;

      }


      const constraints = {

        video: selectedDevice
          ? {
              deviceId: {
                exact: selectedDevice,
              },

              width: {
                ideal: 1280,
              },

              height: {
                ideal: 720,
              },
            }

          : {
              facingMode:
                cameraType === "scene"
                  ? "environment"
                  : "user",

              width: {
                ideal: 1280,
              },

              height: {
                ideal: 720,
              },
            },

        audio: false,

      };


      const stream =
        await navigator.mediaDevices
          .getUserMedia(
            constraints
          );


      streamRef.current =
        stream;


      if (videoRef.current) {

        videoRef.current.srcObject =
          stream;

        await videoRef.current.play();

      }


      setCameraStatus("active");


      if (onStreamReady) {

        onStreamReady(stream);

      }


      // Camera permissions may expose
      // labels after getUserMedia.

      await loadDevices();

    } catch (err) {

      console.error(
        `${title} error:`,
        err
      );

      setCameraStatus("error");

      setError(
        err?.message ||
        "Unable to start camera."
      );

      if (onStreamReady) {

        onStreamReady(null);

      }

    }

  };


  // ==========================================================
  // STOP CAMERA
  // ==========================================================

  const stopCamera = () => {

    if (streamRef.current) {

      streamRef.current
        .getTracks()
        .forEach(
          (track) =>
            track.stop()
        );

      streamRef.current = null;

    }


    if (videoRef.current) {

      videoRef.current.srcObject =
        null;

    }


    setCameraStatus("idle");


    if (onStreamReady) {

      onStreamReady(null);

    }

  };


  // ==========================================================
  // CAMERA ACTIVE / INACTIVE
  // ==========================================================

  useEffect(() => {

    if (isActive) {

      startCamera();

    } else {

      stopCamera();

    }


    return () => {

      stopCamera();

    };

  }, [
    isActive,
    selectedDevice,
  ]);


  // ==========================================================
  // CAPTURE CURRENT FRAME
  // ==========================================================

  const captureFrame = () => {

    const video =
      videoRef.current;


    if (!video) {

      throw new Error(
        `${title}: video element unavailable.`
      );

    }


    if (
      video.readyState <
      HTMLMediaElement.HAVE_CURRENT_DATA
    ) {

      throw new Error(
        `${title}: camera frame is not ready yet.`
      );

    }


    if (
      video.videoWidth === 0 ||
      video.videoHeight === 0
    ) {

      throw new Error(
        `${title}: invalid video dimensions.`
      );

    }


    const canvas =
      document.createElement(
        "canvas"
      );


    canvas.width =
      video.videoWidth;

    canvas.height =
      video.videoHeight;


    const context =
      canvas.getContext("2d");


    context.drawImage(
      video,
      0,
      0,
      canvas.width,
      canvas.height
    );


    return new Promise(
      (resolve, reject) => {

        canvas.toBlob(
          (blob) => {

            if (!blob) {

              reject(
                new Error(
                  `${title}: failed to capture frame.`
                )
              );

              return;

            }


            resolve(blob);

          },

          "image/jpeg",

          0.85
        );

      }
    );

  };


  // ==========================================================
  // EXPOSE FRAME CAPTURE TO PARENT
  // ==========================================================

  useEffect(() => {

    if (onFrameReady) {

      onFrameReady(
        captureFrame
      );

    }

  }, [
    onFrameReady,
    selectedDevice,
    isActive,
  ]);


  // ==========================================================
  // CAMERA SWITCH
  // ==========================================================

  const handleDeviceChange = (
    event
  ) => {

    setSelectedDevice(
      event.target.value
    );

  };


  // ==========================================================
  // RENDER
  // ==========================================================

  return (

    <div className="camera-feed">

      {/* ====================================================
          HEADER
      ==================================================== */}

      <div className="camera-feed-header">

        <div>

          <div className="camera-feed-title">

            {cameraStatus === "active"
              ? <Camera size={18} />
              : <CameraOff size={18} />
            }

            <strong>
              {title}
            </strong>

          </div>

          <span>
            {description}
          </span>

        </div>


        <div
          className={`camera-feed-status ${
            cameraStatus === "active"
              ? "active"
              : ""
          }`}
        >

          <span></span>

          {cameraStatus === "active"
            ? "LIVE"
            : cameraStatus === "starting"
              ? "STARTING"
              : "OFFLINE"}

        </div>

      </div>


      {/* ====================================================
          VIDEO
      ==================================================== */}

      <div className="camera-video-container">

        {isActive ? (

          <video
            ref={videoRef}
            className="camera-video"
            autoPlay
            playsInline
            muted
          />

        ) : (

          <div className="camera-placeholder">

            <CameraOff size={34} />

            <strong>
              Camera inactive
            </strong>

            <span>
              Start camera monitoring
              to activate this stream.
            </span>

          </div>

        )}


        {cameraStatus === "active" && (

          <div className="camera-live-indicator">

            <span></span>

            LIVE

          </div>

        )}

      </div>


      {/* ====================================================
          CAMERA SELECTOR
      ==================================================== */}

      <div className="camera-selector">

        <label>
          Camera Source
        </label>


        <select
          value={selectedDevice}
          onChange={handleDeviceChange}
          disabled={devices.length === 0}
        >

          {devices.length === 0 ? (

            <option value="">
              No camera detected
            </option>

          ) : (

            devices.map(
              (device, index) => (

                <option
                  key={device.deviceId}
                  value={device.deviceId}
                >

                  {device.label ||
                    `Camera ${index + 1}`}

                </option>

              )
            )

          )}

        </select>

      </div>


      {/* ====================================================
          ERROR
      ==================================================== */}

      {error && (

        <div className="camera-error">

          <CameraOff size={16} />

          <span>
            {error}
          </span>

        </div>

      )}

    </div>

  );

}


export default CameraFeed;