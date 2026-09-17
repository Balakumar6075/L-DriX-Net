import {
  Brain,
  Camera,
  Eye,
  Layers,
  Network,
  ScanEye,
  Sparkles,
  Target,
  Zap,
} from "lucide-react";

function About() {
  const architecture = [
    {
      number: "01",
      icon: Camera,
      title: "Driver Camera",
      description:
        "Captures the driver's face and eye region for gaze-related information.",
    },
    {
      number: "02",
      icon: Eye,
      title: "Face Feature Extraction",
      description:
        "LFEM extracts compact facial features from the driver image.",
    },
    {
      number: "03",
      icon: ScanEye,
      title: "Scene Encoding",
      description:
        "The scene encoder extracts visual features from the road environment.",
    },
    {
      number: "04",
      icon: Network,
      title: "ICFM Fusion",
      description:
        "Interactive Cross-Feature Modulation combines face and scene representations.",
    },
    {
      number: "05",
      icon: Layers,
      title: "Temporal Attention",
      description:
        "Attention-based feature processing refines the fused representation.",
    },
    {
      number: "06",
      icon: Target,
      title: "Gaze Prediction",
      description:
        "The prediction head estimates the driver's gaze-related output.",
    },
    {
      number: "07",
      icon: Sparkles,
      title: "Spatial Projection",
      description:
        "The fused representation is projected into a spatial feature map.",
    },
    {
      number: "08",
      icon: Zap,
      title: "Attention Decoder",
      description:
        "The decoder generates a 224 × 224 driver attention heatmap.",
    },
  ];

  const modelDetails = [
    {
      label: "Architecture",
      value: "L-DriX-Net",
    },
    {
      label: "Parameters",
      value: "9.01 Million",
    },
    {
      label: "Input Resolution",
      value: "224 × 224",
    },
    {
      label: "Gaze Output",
      value: "24 Values",
    },
    {
      label: "Attention Map",
      value: "224 × 224",
    },
    {
      label: "Fusion Module",
      value: "ICFM",
    },
  ];

  return (
    <div className="about-page">

      {/* ==================================================
          HEADER
      ================================================== */}

      <section className="about-hero">

        <div className="about-hero-icon">
          <Brain size={38} />
        </div>

        <div>

          <div className="heading-label">
            <Sparkles size={15} />
            RESEARCH PROJECT
          </div>

          <h1>
            L-DriX-Net
          </h1>

          <p>
            A multimodal driver gaze estimation and
            attention prediction framework designed to
            understand where a driver is looking within
            the surrounding road environment.
          </p>

        </div>

      </section>


      {/* ==================================================
          OVERVIEW
      ================================================== */}

      <section className="about-section">

        <div className="about-section-heading">

          <div className="section-number">
            01
          </div>

          <div>

            <h2>
              Project Overview
            </h2>

            <p>
              Understanding driver visual attention
            </p>

          </div>

        </div>


        <div className="overview-grid">

          <div className="overview-card">

            <div className="overview-icon">
              <Eye size={23} />
            </div>

            <h3>
              Gaze Estimation
            </h3>

            <p>
              L-DriX-Net processes driver-related visual
              information to estimate gaze characteristics
              and viewing direction.
            </p>

          </div>


          <div className="overview-card">

            <div className="overview-icon">
              <Camera size={23} />
            </div>

            <h3>
              Dual-Camera Input
            </h3>

            <p>
              The proposed interface is designed around
              two synchronized camera streams: one facing
              the driver and another observing the road.
            </p>

          </div>


          <div className="overview-card">

            <div className="overview-icon">
              <Target size={23} />
            </div>

            <h3>
              Attention Prediction
            </h3>

            <p>
              The system generates a spatial attention
              representation showing regions of the scene
              associated with predicted driver attention.
            </p>

          </div>

        </div>

      </section>


      {/* ==================================================
          ARCHITECTURE
      ================================================== */}

      <section className="about-section">

        <div className="about-section-heading">

          <div className="section-number">
            02
          </div>

          <div>

            <h2>
              Model Architecture
            </h2>

            <p>
              Multimodal processing pipeline
            </p>

          </div>

        </div>


        <div className="architecture-flow">

          {architecture.map((item, index) => {

            const Icon = item.icon;

            return (
              <div
                className="architecture-wrapper"
                key={item.number}
              >

                <div className="architecture-card">

                  <div className="architecture-top">

                    <span className="architecture-number">
                      {item.number}
                    </span>

                    <div className="architecture-icon">
                      <Icon size={21} />
                    </div>

                  </div>


                  <h3>
                    {item.title}
                  </h3>


                  <p>
                    {item.description}
                  </p>

                </div>


                {index < architecture.length - 1 && (
                  <div className="architecture-arrow">
                    →
                  </div>
                )}

              </div>
            );

          })}

        </div>

      </section>


      {/* ==================================================
          ARCHITECTURE SUMMARY
      ================================================== */}

      <section className="architecture-summary">

        <div className="summary-header">

          <div className="summary-icon">
            <Network size={23} />
          </div>

          <div>

            <h3>
              L-DriX-Net Data Flow
            </h3>

            <p>
              From camera input to attention prediction
            </p>

          </div>

        </div>


        <div className="summary-flow">

          <span>
            Driver Image
          </span>

          <b>→</b>

          <span>
            LFEM
          </span>

          <b>→</b>

          <span>
            Face Features
          </span>

          <b>→</b>

          <span>
            ICFM
          </span>

          <b>→</b>

          <span>
            Temporal Attention
          </span>

          <b>→</b>

          <span>
            Prediction
          </span>

        </div>


        <div className="summary-flow secondary">

          <span>
            Scene Image
          </span>

          <b>→</b>

          <span>
            Scene Encoder
          </span>

          <b>→</b>

          <span>
            Scene Features
          </span>

          <b>→</b>

          <span>
            ICFM
          </span>

          <b>→</b>

          <span>
            Spatial Projection
          </span>

          <b>→</b>

          <span>
            Heatmap
          </span>

        </div>

      </section>


      {/* ==================================================
          MODEL DETAILS
      ================================================== */}

      <section className="about-section">

        <div className="about-section-heading">

          <div className="section-number">
            03
          </div>

          <div>

            <h2>
              Model Specifications
            </h2>

            <p>
              Current implementation details
            </p>

          </div>

        </div>


        <div className="model-details">

          {modelDetails.map((item) => (

            <div
              className="model-detail"
              key={item.label}
            >

              <span>
                {item.label}
              </span>

              <strong>
                {item.value}
              </strong>

            </div>

          ))}

        </div>

      </section>


      {/* ==================================================
          TECHNOLOGY
      ================================================== */}

      <section className="technology-card">

        <div className="technology-heading">

          <div className="technology-icon">
            <Zap size={22} />
          </div>

          <div>

            <h2>
              Technology Stack
            </h2>

            <p>
              Components used to build the system
            </p>

          </div>

        </div>


        <div className="technology-list">

          <div className="technology-item">

            <strong>
              React
            </strong>

            <span>
              Frontend interface
            </span>

          </div>


          <div className="technology-item">

            <strong>
              FastAPI
            </strong>

            <span>
              Backend API
            </span>

          </div>


          <div className="technology-item">

            <strong>
              PyTorch
            </strong>

            <span>
              Deep learning inference
            </span>

          </div>


          <div className="technology-item">

            <strong>
              L-DriX-Net
            </strong>

            <span>
              Gaze estimation model
            </span>

          </div>

        </div>

      </section>


      {/* ==================================================
          FOOTER
      ================================================== */}

      <section className="about-footer">

        <div className="footer-brain">
          <Brain size={25} />
        </div>

        <div>

          <strong>
            L-DriX-Net
          </strong>

          <span>
            Driver Gaze Estimation & Attention Analysis
          </span>

        </div>

      </section>

    </div>
  );
}

export default About;