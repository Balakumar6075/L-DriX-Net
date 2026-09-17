import {
  LayoutDashboard,
  ScanEye,
  History,
  Info,
  Brain,
} from "lucide-react";


function Sidebar({
  activePage,
  setActivePage,
}) {

  const menuItems = [
    {
      id: "dashboard",
      label: "Dashboard",
      icon: LayoutDashboard,
    },
    {
      id: "prediction",
      label: "Prediction",
      icon: ScanEye,
    },
    {
      id: "history",
      label: "History",
      icon: History,
    },
    {
      id: "about",
      label: "About",
      icon: Info,
    },
  ];


  return (
    <aside className="sidebar">

      {/* ==================================================
          LOGO
      =================================================== */}

      <div className="sidebar-logo">

        <div className="logo-icon">
          <Brain size={25} />
        </div>

        <div className="logo-text">

          <h2>
            L-DriX-Net
          </h2>

          <span>
            Gaze Estimation
          </span>

        </div>

      </div>


      {/* ==================================================
          NAVIGATION
      =================================================== */}

      <nav className="sidebar-nav">

        <p className="nav-title">
          MAIN MENU
        </p>


        {menuItems.map((item) => {

          const Icon = item.icon;

          return (

            <button
              key={item.id}
              type="button"
              className={`nav-item ${
                activePage === item.id
                  ? "active"
                  : ""
              }`}
              onClick={() =>
                setActivePage(item.id)
              }
            >

              <Icon size={20} />

              <span>
                {item.label}
              </span>

            </button>

          );

        })}

      </nav>


      {/* ==================================================
          MODEL INFORMATION
      =================================================== */}

      <div className="sidebar-bottom">

        <div className="model-status">

          <div className="status-dot"></div>

          <div>

            <strong>
              Model Status
            </strong>

            <span>
              Ready for prediction
            </span>

          </div>

        </div>


        <div className="model-info">

          <span>
            Model
          </span>

          <strong>
            L-DriX-Net
          </strong>

        </div>


        <div className="model-info">

          <span>
            Parameters
          </span>

          <strong>
            9.01M
          </strong>

        </div>

      </div>

    </aside>
  );
}


export default Sidebar;