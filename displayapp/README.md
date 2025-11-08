# 🚁 FPV Drone Detection - Display Application

Real-time web dashboard for visualizing and analyzing FPV drone detections with intelligent threat assessment based on distance, direction, and drone heading.

## 🎯 Overview

This Next.js application provides a comprehensive visualization system for drone detection data, featuring:

- **Interactive Map**: Real-time drone positions with camera locations using Leaflet
- **Intelligent Threat Analysis**: Multi-factor danger assessment combining:
  - Distance to user
  - Drone heading (direction of travel)
  - Angular deviation from threat vector
  - Estimated time of arrival (ETA)
- **Color-Coded Alerts**: Visual threat levels from 🔴 Critical to 🟢 Safe
- **Detection Management**: Persistent blacklist system for filtering detections

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ and npm
- Python 3.x (for test scripts)

### Installation

```bash
npm install
```

### Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## 📡 API Endpoints

### POST `/api/detections`

Submit a new drone detection.

**Request Body:**
```json
{
  "filename": "drone_001.jpg",
  "confidence": 0.95,
  "bbox": {
    "x": 100,
    "y": 200,
    "width": 150,
    "height": 120
  },
  "camera_id": "CAM_03",
  "camera_location": {
    "lat": 48.8606,
    "lng": 2.3376,
    "name": "Louvre"
  },
  "image_dimensions": {
    "width": 1920,
    "height": 1080
  },
  "drone_heading": 174.9,
  "drone_location": {
    "lat": 48.857,
    "lng": 2.338
  }
}
```

**Optional Fields:**
- `drone_heading`: Drone's heading/direction in degrees (0-360°). If not provided, worst-case scenario (heading toward user) is assumed.
- `drone_location`: GPS coordinates of the drone. If not provided, position is calculated from camera location and bbox.

**Response:**
```json
{
  "id": "det_1699459200000_abc123",
  "message": "Detection stored successfully"
}
```

### GET `/api/detections`

Retrieve all stored detections (max 100 recent).

**Response:**
```json
{
  "detections": [
    {
      "id": "det_1699459200000_abc123",
      "timestamp": 1699459200000,
      "filename": "drone_001.jpg",
      "confidence": 0.95,
      "bbox": { "x": 100, "y": 200, "width": 150, "height": 120 },
      "camera_id": "CAM_03",
      "drone_heading": 174.9,
      "drone_location": { "lat": 48.857, "lng": 2.338 }
    }
  ],
  "count": 1
}
```

### DELETE `/api/detections/:id`

Delete a specific detection by ID.

**Response:**
```json
{
  "message": "Detection deleted successfully"
}
```

## 🗺️ Map Features

### Camera Locations (Mocked)

- 📷 **Tour Eiffel**: `CAM_01` (48.8566, 2.3522)
- 📷 **Arc de Triomphe**: `CAM_02` (48.8738, 2.2950)
- 📷 **Louvre**: `CAM_03` (48.8606, 2.3376)
- 📷 **Notre-Dame**: `CAM_04` (48.8529, 2.3499)

### User Position (Mocked)

- 👤 **User Location**: (48.8584, 2.3448)

### Drone Visualization

Each detected drone displays:
- **Circle**: Color-coded by threat level
- **Camera Line**: Dashed line from camera to drone
- **Direction Arrow**: Shows drone heading (direction of travel)
- **Hover Tooltip**: Quick stats (distance, heading, threat level)
- **Click Popup**: Detailed analysis with delete button

## 🎨 Threat Level System

### Angular Thresholds

The system evaluates drone threat based on **heading deviation** from direct approach:

- 🔴 **0-20°**: Direct approach - CRITICAL/DANGER
- 🟠 **20-40°**: Approaching - WARNING
- 🟡 **40-70°**: Oblique - CAUTION
- 🟢 **70°+**: Lateral/Moving away - SAFE

### Distance Integration

Color intensity also depends on distance to user:

**Direct Approach (0-40° deviation):**
- <150m → 🔴 Critical (dark red)
- 150-300m → 🔴 Danger (red)
- 300-500m → 🟠 Warning (orange)
- >500m → 🟡/🟢 Safe (yellow/green)

**Oblique/Lateral (40°+ deviation):**
- <100m → 🔴 Danger (very close)
- 100-250m → 🟠 Warning (medium)
- 250-500m → 🟢 Safe (green light)
- >500m → 🟢 Safe (green)

**Moving Away:**
- <100m → 🟠 Warning (close even if retreating)
- >100m → 🟢 Safe (green)

## 🧪 Testing

### Test Scripts

Located in parent directory (`../`):

**1. Distance + Angle Test**
```bash
python test_distance_angle.py
```

Sends 9 test scenarios combining various distances (100m-600m) and angles (0°-180°) to validate threat assessment logic.

**2. Drone Heading Test**
```bash
python test_drone_caps.py
```

Tests arbitrary drone headings independent of camera detection angle.

### Manual Testing with Postman/cURL

```bash
curl -X POST http://localhost:3000/api/detections \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "test.jpg",
    "confidence": 0.92,
    "bbox": {"x": 500, "y": 300, "width": 200, "height": 180},
    "camera_id": "CAM_03",
    "drone_heading": 115,
    "drone_location": {"lat": 48.8570, "lng": 2.3381}
  }'
```

## 🧮 Core Algorithms

### 1. Direction Estimation (`estimateDirection`)

Calculates drone bearing from camera based on bbox position in image frame.

**Formula**: `bearing = atan2(offsetX, -offsetY) * (180/π)`

### 2. Bearing Calculation (`calculateBearingToUser`)

Computes geodesic bearing from drone position to user using Haversine formula.

### 3. Danger Evaluation (`evaluateDanger`)

Multi-factor threat assessment:

```typescript
evaluateDanger(
  dronePosition: {lat, lng},
  userPosition: {lat, lng},
  droneSpeed: number,  // m/s
  droneHeading: number // degrees (0-360°)
)
```

**Returns:**
- `level`: critical | danger | warning | low | safe
- `color`: Hex color code
- `distanceToUser`: Distance in meters
- `angleDifference`: Heading deviation from threat vector
- `eta`: Estimated time of arrival (if approaching)
- `threatAnalysis`: Human-readable description

## 📂 Project Structure

```
displayapp/
├── app/
│   ├── api/
│   │   └── detections/
│   │       ├── route.ts          # GET/POST endpoints
│   │       └── [id]/route.ts     # DELETE endpoint
│   ├── layout.tsx                # Root layout
│   └── page.tsx                  # Main dashboard
├── components/
│   ├── DetectionDashboard.tsx    # Tab navigation
│   ├── DetectionList.tsx         # Table view
│   └── DetectionMap.tsx          # Interactive map
├── lib/
│   └── droneEstimations.ts       # Core algorithms
└── public/
    └── ...
```

## 🔧 Technology Stack

- **Framework**: Next.js 16.0.1 (React 19)
- **Language**: TypeScript (strict mode)
- **Styling**: Tailwind CSS 4.1.9
- **Maps**: Leaflet 1.9+ with react-leaflet
- **Tiles**: CartoDB Positron No Labels
- **State**: React hooks + localStorage (blacklist)
- **API**: Next.js Route Handlers

## 🎛️ Configuration

### User Position

Edit in `components/DetectionMap.tsx`:

```typescript
const USER_POSITION = { 
  lat: 48.8584, 
  lng: 2.3448, 
  name: "Your Location" 
}
```

### Threat Thresholds

Edit in `lib/droneEstimations.ts`:

```typescript
const THREAT_ANGLE_CONFIG = {
  direct: 20,       // 0-20° = Critical/Danger
  approaching: 40,  // 20-40° = Warning
  oblique: 70,      // 40-70° = Caution
  lateral: 120,     // 70-120° = Safe
}
```

### Polling Interval

Default: 2000ms. Edit in `components/DetectionDashboard.tsx`:

```typescript
const POLLING_INTERVAL = 2000 // milliseconds
```

## 📊 Performance

- **In-Memory Storage**: Max 100 recent detections
- **Auto-Cleanup**: Old detections pruned automatically
- **Client Polling**: 2-second intervals to reduce server load
- **Efficient Rendering**: Only updates changed detections on map

## 🔐 Security Notes

⚠️ **This is a prototype/demo application:**

- No authentication/authorization
- In-memory storage (data lost on restart)
- CORS enabled for development
- Not production-ready

## 📝 License

MIT

## 🤝 Contributing

This is a hackathon prototype. For production use, consider:

- Database persistence (PostgreSQL/MongoDB)
- User authentication (NextAuth.js)
- WebSocket real-time updates
- Production-grade error handling
- Rate limiting
- Input validation/sanitization
