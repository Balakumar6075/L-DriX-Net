const API_BASE_URL =
  import.meta.env.VITE_API_URL ||
  "http://127.0.0.1:8000";

export async function checkBackendStatus() {
  try {
    const response = await fetch(
      `${API_BASE_URL}/health`
    );

    if (!response.ok) {
      throw new Error(
        "Backend health check failed."
      );
    }

    return await response.json();
  } catch (error) {
    console.error(
      "Backend status error:",
      error
    );

    return {
      status: "offline",
      message: "Backend unavailable",
    };
  }
}

export async function predictGaze(
  driverFrame,
  sceneFrame
) {
  const formData = new FormData();

  formData.append(
    "driver_image",
    driverFrame,
    "driver.jpg"
  );

  formData.append(
    "scene_image",
    sceneFrame,
    "scene.jpg"
  );

  const response = await fetch(
    `${API_BASE_URL}/predict`,
    {
      method: "POST",
      body: formData,
    }
  );

  if (!response.ok) {
    const errorText =
      await response.text();

    throw new Error(
      errorText ||
        `Prediction request failed (${response.status})`
    );
  }

  return await response.json();
}

export async function sendFrame(
  driverFrame,
  sceneFrame
) {
  return predictGaze(
    driverFrame,
    sceneFrame
  );
}