import {
  BarChart3,
  Target,
  Activity,
  TrendingDown,
} from "lucide-react";

function Metrics({ prediction }) {
  /*
    These values currently represent the evaluation
    metrics from the existing L-DriX-Net project.

    Later, the FastAPI backend can return live
    inference/evaluation values.
  */

  const metrics = [
    {
      name: "MAE",
      value: prediction ? "9.9573" : "--",
      description: "Mean Absolute Error",
      icon: Target,
      className: "metric-blue",
    },
    {
      name: "RMSE",
      value: prediction ? "45.1664" : "--",
      description: "Root Mean Square Error",
      icon: BarChart3,
      className: "metric-purple",
    },
    {
      name: "KL Loss",
      value: prediction ? "0.0012" : "--",
      description: "Kullback-Leibler Divergence",
      icon: TrendingDown,
      className: "metric-green",
    },
    {
      name: "NCC",
      value: prediction ? "0.2524" : "--",
      description: "Normalized Cross-Correlation",
      icon: Activity,
      className: "metric-orange",
    },
  ];

  return (
    <div className="metrics-grid">

      {metrics.map((metric) => {
        const Icon = metric.icon;

        return (
          <div
            className={`metric-card ${metric.className}`}
            key={metric.name}
          >

            {/* Icon */}
            <div className="metric-top">

              <div className="metric-icon">
                <Icon size={21} />
              </div>

              <span className="metric-name">
                {metric.name}
              </span>

            </div>


            {/* Value */}
            <div className="metric-value">

              <strong>
                {metric.value}
              </strong>

            </div>


            {/* Description */}
            <div className="metric-description">

              {metric.description}

            </div>


            {/* Status */}
            <div className="metric-status">

              <span
                className={
                  prediction
                    ? "metric-dot active"
                    : "metric-dot"
                }
              />

              {prediction
                ? "Available"
                : "Waiting for prediction"}

            </div>

          </div>
        );
      })}

    </div>
  );
}

export default Metrics;