import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import GUI from 'lil-gui';
import './style.css';

// --- CONFIGURATION ---
interface CableConfig {
  sensors: number[]; // Array of temperatures
  level?: number;
  bottomOffset?: number;
  topOffset?: number;
}

interface RingConfig {
  diameter: number;
  cableCount?: number;
  sensorsPerCable?: number;
  cables?: CableConfig[];
}

interface SiloConfig {
  silo: {
    radius: number;
    height: number;
    grainHeight: number;
  };
  scale?: {
    min: number;
    max: number;
  };
  horizontal?: number;
  vertical?: number;
  rings: RingConfig[];
}

let SILO_RADIUS = 5;
let SILO_HEIGHT = 15;
let GRAIN_HEIGHT = 12;
let TEMPERATURE_MIN = 10;
let TEMPERATURE_MAX = 50;

// --- SHADERS ---
const vertexShader = `
  varying float vTemperature;
  varying vec3 vPosition;
  
  uniform vec4 sensors[100]; // Increased to 100 points
  uniform int sensorCount;
  uniform float pH; // IDW Horizontal power factor
  uniform float pV; // IDW Vertical power factor
  uniform vec3 clippingPlaneNormal;
  uniform float clippingPlaneConstant;

  void main() {
    vPosition = position;
    
    // IDW Interpolation
    float weightSum = 0.0;
    float tempSum = 0.0;
    bool exactMatch = false;

    for (int i = 0; i < 100; i++) {
      if (i >= sensorCount) break;
      
      float dist = distance(position, sensors[i].xyz);
      if (dist < 0.01) {
        vTemperature = sensors[i].w;
        exactMatch = true;
        break;
      }
      
      float distH = distance(position.xz, sensors[i].xz);
      float distV = abs(position.y - sensors[i].y);
      float weight = 1.0 / (pow(distH + 0.001, pH) + pow(distV + 0.001, pV));
      
      weightSum += weight;
      tempSum += sensors[i].w * weight;
    }

    if (!exactMatch) {
      vTemperature = (weightSum > 0.0) ? (tempSum / weightSum) : 25.0;
    }

    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mvPosition;
    gl_PointSize = 4.0;
    
    // Pass clipping information to fragment shader
    vPosition = (modelMatrix * vec4(position, 1.0)).xyz;
  }
`;

const fragmentShader = `
  varying float vTemperature;
  varying vec3 vPosition;
  
  uniform float tMin;
  uniform float tMax;
  uniform vec3 clippingPlaneNormal;
  uniform float clippingPlaneConstant;

  // Heatmap Palette (Jet-like)
  vec3 heatmapColor(float t) {
    float v = clamp((t - tMin) / (tMax - tMin), 0.0, 1.0);
    vec3 c1 = vec3(0.0, 0.0, 1.0); // Blue
    vec3 c2 = vec3(0.0, 1.0, 1.0); // Cyan
    vec3 c3 = vec3(0.0, 1.0, 0.0); // Green
    vec3 c4 = vec3(1.0, 1.0, 0.0); // Yellow
    vec3 c5 = vec3(1.0, 0.0, 0.0); // Red
    
    if (v < 0.25) return mix(c1, c2, v * 4.0);
    if (v < 0.5) return mix(c2, c3, (v - 0.25) * 4.0);
    if (v < 0.75) return mix(c3, c4, (v - 0.5) * 4.0);
    return mix(c4, c5, (v - 0.75) * 4.0);
  }

  void main() {
    // Clipping
    if (dot(vPosition, clippingPlaneNormal) > clippingPlaneConstant) {
        discard;
    }

    gl_FragColor = vec4(heatmapColor(vTemperature), 0.8);
  }
`;

// --- APP SETUP ---
class SiloApp {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: OrbitControls;
  sensors: { pos: THREE.Vector3, temp: number, baseTemp: number }[] = [];
  cableLevels: { pos: THREE.Vector2, height: number }[] = [];
  grainPoints: THREE.Points | null = null;
  clippingPlane: THREE.Plane;
  gui: GUI = new GUI();

  params = {
    slicing: 0,
    showSilo: true,
    showSensors: true,
    smoothnessH: 2.0,
    smoothnessV: 2.0,
    animateSensors: true,
    autoRotate: false,
    loadConfig: () => {
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = '.json';
      input.onchange = (e: any) => {
        const file = e.target.files[0];
        const reader = new FileReader();
        reader.onload = (event: any) => {
          try {
            const config = JSON.parse(event.target.result);
            this.applyConfig(config);
          } catch (err) {
            console.error("Erro ao ler JSON:", err);
            alert("Erro ao ler o arquivo JSON.");
          }
        };
        reader.readAsText(file);
      };
      input.click();
    }
  };

  constructor() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0a0a0a);

    this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    this.camera.position.set(15, 15, 15);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.localClippingEnabled = true;
    document.getElementById('app')?.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.enablePan = true;
    this.controls.autoRotate = this.params.autoRotate;
    this.controls.autoRotateSpeed = 1.0;
    this.controls.maxPolarAngle = Math.PI / 2; // Limit rotation to 90 degrees (prevents looking from below)

    this.clippingPlane = new THREE.Plane(new THREE.Vector3(1, 0, 0), 0);

    this.initLights();
    this.initSilo();
    this.initSensors();
    this.initGrainMass();
    this.initGUI();

    window.addEventListener('resize', this.onResize.bind(this));
    this.animate();

    // Default Configuration Loading
    this.loadDefaultConfig();
  }

  async loadDefaultConfig() {
    try {
      const response = await fetch('/config.json');
      if (response.ok) {
        const config = await response.json();
        this.applyConfig(config);
      }
    } catch (err) {
      console.warn("Could not load default config.json. Using defaults.", err);
    }
  }

  applyConfig(config: SiloConfig) {
    // Update globals used in geometry creation
    SILO_RADIUS = config.silo.radius;
    SILO_HEIGHT = config.silo.height;
    GRAIN_HEIGHT = config.silo.grainHeight;

    if (config.scale) {
      TEMPERATURE_MIN = config.scale.min;
      TEMPERATURE_MAX = config.scale.max;
    }

    if (config.horizontal !== undefined) {
      this.params.smoothnessH = config.horizontal;
    }
    if (config.vertical !== undefined) {
      this.params.smoothnessV = config.vertical;
    }

    // Reset Scene
    while (this.scene.children.length > 0) {
      this.scene.remove(this.scene.children[0]);
    }

    this.initLights();
    this.initSilo();

    this.sensors = [];
    this.cableLevels = [];

    config.rings.forEach(ring => {
      const radius = ring.diameter / 2;

      if (ring.cables && ring.cables.length > 0) {
        // Use explicit cables from JSON
        ring.cables.forEach((cable, i) => {
          const angle = (i / ring.cables!.length) * Math.PI * 2;
          const x = Math.cos(angle) * radius;
          const z = Math.sin(angle) * radius;

          const levelIndex = cable.level !== undefined ? cable.level : cable.sensors.length;
          const bottomOffset = cable.bottomOffset !== undefined ? cable.bottomOffset : 0;
          const topOffset = cable.topOffset !== undefined ? cable.topOffset : 0;

          const roofHeight = this.getRoofHeight(x, z);
          const sensorsTop = roofHeight - topOffset;

          // Calculate height based on sensor index
          // level 1 = first sensor, level N = Nth sensor
          const sensorCount = cable.sensors.length;
          let visualHeight = 0;
          if (levelIndex > 0) {
            const ratio = Math.min(levelIndex - 1, sensorCount - 1) / Math.max(sensorCount - 1, 1);
            visualHeight = bottomOffset + ratio * (sensorsTop - bottomOffset);
            visualHeight += 0.3; // "Ligeiramente acima" offset
          }

          this.cableLevels.push({ pos: new THREE.Vector2(x, z), height: visualHeight });
          this.addCable(x, z, cable.sensors.length, cable.sensors, bottomOffset, topOffset);
        });
      } else if (ring.cableCount && ring.sensorsPerCable) {
        // Fallback to auto-generation
        for (let i = 0; i < ring.cableCount; i++) {
          const angle = (i / ring.cableCount) * Math.PI * 2;
          const x = Math.cos(angle) * radius;
          const z = Math.sin(angle) * radius;

          this.cableLevels.push({ pos: new THREE.Vector2(x, z), height: GRAIN_HEIGHT });
          this.addCable(x, z, ring.sensorsPerCable);
        }
      }
    });

    this.initGrainMass();

    // Update UI constraints if needed (lil-gui doesn't update min/max easily without recreation)
    this.params.slicing = SILO_RADIUS;
    this.clippingPlane.constant = SILO_RADIUS;

    // Update shader uniforms for range
    if (this.grainPoints) {
      (this.grainPoints.material as THREE.ShaderMaterial).uniforms.tMin.value = TEMPERATURE_MIN;
      (this.grainPoints.material as THREE.ShaderMaterial).uniforms.tMax.value = TEMPERATURE_MAX;
      (this.grainPoints.material as THREE.ShaderMaterial).uniforms.pH.value = this.params.smoothnessH;
      (this.grainPoints.material as THREE.ShaderMaterial).uniforms.pV.value = this.params.smoothnessV;
    }
  }

  addCable(x: number, z: number, count: number, temperatures?: number[], bottomOffset?: number, topOffset?: number) {
    const sensorGeo = new THREE.SphereGeometry(0.1, 8, 8);
    const sensorMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const bOffset = bottomOffset !== undefined ? bottomOffset : 0;
    const tOffset = topOffset !== undefined ? topOffset : 0;

    // Roof Height calculation
    const roofHeightAtPos = this.getRoofHeight(x, z);

    const cableTop = roofHeightAtPos;
    const sensorsTop = roofHeightAtPos - tOffset;

    // Vertical line for cable
    const points = [new THREE.Vector3(x, 0, z), new THREE.Vector3(x, cableTop, z)];
    const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
    const line = new THREE.Line(lineGeo, new THREE.LineBasicMaterial({ color: 0xcccccc, transparent: true, opacity: 0.5 }));
    line.name = 'cable';
    this.scene.add(line);

    for (let s = 0; s < count; s++) {
      const y = bOffset + (s / (count - 1)) * (sensorsTop - bOffset);
      const pos = new THREE.Vector3(x, y, z);

      // Use provided temp or initial random
      const temp = (temperatures && temperatures[s] !== undefined)
        ? temperatures[s]
        : TEMPERATURE_MIN + Math.random() * (TEMPERATURE_MAX - TEMPERATURE_MIN);

      this.sensors.push({ pos, temp, baseTemp: temp });

      const sensorMesh = new THREE.Mesh(sensorGeo, sensorMaterial);
      sensorMesh.position.copy(pos);
      sensorMesh.name = 'sensor';
      this.scene.add(sensorMesh);
    }
  }

  getRoofHeight(x: number, z: number): number {
    const dist = Math.sqrt(x * x + z * z);
    const roofTipHeight = 3;
    const roofRadius = SILO_RADIUS * 1.05;
    return SILO_HEIGHT + roofTipHeight * (1 - Math.min(dist / roofRadius, 1));
  }

  initLights() {
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    this.scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 1);
    dirLight.position.set(10, 20, 10);
    this.scene.add(dirLight);
  }

  initSilo() {
    const group = new THREE.Group();
    group.name = 'silo';

    // Cylinder
    const cylinderGeo = new THREE.CylinderGeometry(SILO_RADIUS, SILO_RADIUS, SILO_HEIGHT, 32, 1, true);
    const material = new THREE.MeshStandardMaterial({
      color: 0x888888,
      side: THREE.BackSide,
      transparent: true,
      opacity: 0.3,
      clippingPlanes: [this.clippingPlane]
    });
    const cylinder = new THREE.Mesh(cylinderGeo, material);
    cylinder.position.y = SILO_HEIGHT / 2;
    group.add(cylinder);

    // Roof (Cone)
    const coneGeo = new THREE.ConeGeometry(SILO_RADIUS * 1.05, 3, 32, 1, true);
    const cone = new THREE.Mesh(coneGeo, material);
    cone.position.y = SILO_HEIGHT + 1.5;
    group.add(cone);

    // Floor
    const floorGeo = new THREE.CircleGeometry(SILO_RADIUS, 32);
    const floorMaterial = new THREE.MeshStandardMaterial({ color: 0x444444, side: THREE.DoubleSide });
    const floor = new THREE.Mesh(floorGeo, floorMaterial);
    floor.rotation.x = -Math.PI / 2;
    group.add(floor);

    this.scene.add(group);
  }

  initSensors() {
    this.cableLevels = [];
    // Initial default setup
    const SENSOR_CABLES = 5;
    const SENSORS_PER_CABLE = 10;

    // 5 Periphery Cables (10 sensors each)
    for (let c = 0; c < SENSOR_CABLES; c++) {
      const angle = (c / SENSOR_CABLES) * Math.PI * 2;
      const radius = SILO_RADIUS * 0.7; // Closer to wall
      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;

      this.cableLevels.push({ pos: new THREE.Vector2(x, z), height: GRAIN_HEIGHT });
      this.addCable(x, z, SENSORS_PER_CABLE);
    }

    // 1 Central Cable (11 sensors)
    this.cableLevels.push({ pos: new THREE.Vector2(0, 0), height: GRAIN_HEIGHT });
    this.addCable(0, 0, 11);
  }

  initGrainMass() {
    const count = 50000;
    const positions = new Float32Array(count * 3);

    for (let i = 0; i < count; i++) {
      // Random point within cylinder
      const r = Math.sqrt(Math.random()) * SILO_RADIUS;
      const theta = Math.random() * Math.PI * 2;
      const x = r * Math.cos(theta);
      const z = r * Math.sin(theta);

      // Calculate max height at this (x,z) using IDW interpolation of cable levels
      let weightSum = 0;
      let heightSum = 0;
      let yMax = GRAIN_HEIGHT;

      if (this.cableLevels.length > 0) {
        let exactMatch = false;
        for (const cl of this.cableLevels) {
          const dist = cl.pos.distanceTo(new THREE.Vector2(x, z));
          if (dist < 0.01) {
            yMax = cl.height;
            exactMatch = true;
            break;
          }
          const weight = 1.0 / Math.pow(dist, 2.0); // Fixed power for surface smoothness
          weightSum += weight;
          heightSum += cl.height * weight;
        }
        if (!exactMatch) {
          yMax = heightSum / weightSum;
        }
      }

      const y = Math.random() * yMax;

      positions[i * 3] = x;
      positions[i * 3 + 1] = y;
      positions[i * 3 + 2] = z;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const sensorData = new Float32Array(100 * 4); // Match shader uniform size
    this.updateSensorUniforms(sensorData);

    const material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        sensors: { value: sensorData },
        sensorCount: { value: this.sensors.length },
        pH: { value: this.params.smoothnessH },
        pV: { value: this.params.smoothnessV },
        tMin: { value: TEMPERATURE_MIN },
        tMax: { value: TEMPERATURE_MAX },
        clippingPlaneNormal: { value: this.clippingPlane.normal },
        clippingPlaneConstant: { value: this.clippingPlane.constant }
      },
      transparent: true
    });

    this.grainPoints = new THREE.Points(geometry, material);
    this.scene.add(this.grainPoints);
  }

  updateSensorUniforms(buffer: Float32Array) {
    this.sensors.forEach((s, i) => {
      if (i < 100) {
        buffer[i * 4] = s.pos.x;
        buffer[i * 4 + 1] = s.pos.y;
        buffer[i * 4 + 2] = s.pos.z;
        buffer[i * 4 + 3] = s.temp;
      }
    });
  }

  initGUI() {
    this.gui = new GUI();
    const folder = this.gui.addFolder('Controles');

    folder.add(this.params, 'slicing', -SILO_RADIUS, SILO_RADIUS).name('Fatiar (X)').onChange((v: number) => {
      this.clippingPlane.constant = v;
      if (this.grainPoints) {
        (this.grainPoints.material as THREE.ShaderMaterial).uniforms.clippingPlaneConstant.value = v;
      }
    });

    folder.add(this.params, 'showSilo').name('Mostrar Silo').onChange((v: boolean) => {
      const silo = this.scene.getObjectByName('silo');
      if (silo) silo.visible = v;
    });

    folder.add(this.params, 'showSensors').name('Mostrar Sensores').onChange((v: boolean) => {
      this.scene.traverse((obj) => {
        if (obj.name === 'sensor' || obj.name === 'cable') obj.visible = v;
      });
    });

    folder.add(this.params, 'smoothnessH', 0.0, 20.0).name('Suavidade Horiz.').onChange((v: number) => {
      if (this.grainPoints) {
        (this.grainPoints.material as THREE.ShaderMaterial).uniforms.pH.value = v;
      }
    });

    folder.add(this.params, 'smoothnessV', 0.0, 20.0).name('Suavidade Vert.').onChange((v: number) => {
      if (this.grainPoints) {
        (this.grainPoints.material as THREE.ShaderMaterial).uniforms.pV.value = v;
      }
    });

    folder.add(this.params, 'animateSensors').name('Animar Temperaturas');
    folder.add(this.params, 'autoRotate').name('Giro Automático').onChange((v: boolean) => {
      this.controls.autoRotate = v;
    });
    folder.add(this.params, 'loadConfig').name('Carregar JSON Config');

    folder.open();
  }

  onResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  }

  animate() {
    requestAnimationFrame(this.animate.bind(this));

    if (this.params.animateSensors) {
      const time = performance.now() * 0.001;
      this.sensors.forEach((s, i) => {
        // Fluctuate +/- 1.0 degree around baseTemp
        s.temp = s.baseTemp + Math.sin(time + i * 0.1) * 1.5;
      });

      if (this.grainPoints) {
        this.updateSensorUniforms((this.grainPoints.material as THREE.ShaderMaterial).uniforms.sensors.value);
      }
    } else {
      // Reset to exact JSON values when animation is OFF
      let changed = false;
      this.sensors.forEach(s => {
        if (s.temp !== s.baseTemp) {
          s.temp = s.baseTemp;
          changed = true;
        }
      });
      if (changed && this.grainPoints) {
        this.updateSensorUniforms((this.grainPoints.material as THREE.ShaderMaterial).uniforms.sensors.value);
      }
    }

    // Restrict pan to Y axis only (keep centered horizontally)
    this.controls.target.x = 0;
    this.controls.target.z = 0;

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

new SiloApp();
