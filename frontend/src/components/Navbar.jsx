import {
  Bell,
  UserCircle,
  Activity,
} from "lucide-react";


function Navbar() {

  return (
    <header className="navbar">


      {/* ==================================================
          LEFT SIDE
      =================================================== */}

      <div className="navbar-left">

        <div className="page-indicator">

          <Activity size={19} />

        </div>


        <div>

          <h1>
            Driver Attention Analysis
          </h1>

          <p>
            Real-time gaze estimation and attention prediction
          </p>

        </div>

      </div>


      {/* ==================================================
          RIGHT SIDE
      =================================================== */}

      <div className="navbar-right">


        {/* SYSTEM STATUS */}

        <div className="system-status">

          <span className="status-dot"></span>

          System Online

        </div>


        {/* NOTIFICATION */}

        <button
          type="button"
          className="icon-button"
          title="Notifications"
        >

          <Bell size={20} />

        </button>


        {/* USER / RESEARCH MODE */}

        <div className="user-profile">

          <UserCircle size={34} />

          <div>

            <strong>
              Research Mode
            </strong>

            <span>
              L-DriX-Net
            </span>

          </div>

        </div>

      </div>

    </header>
  );
}


export default Navbar;