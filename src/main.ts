import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js';
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
  varying float vViewZ;
  
  attribute float size; // Added for grain variation
  
  uniform vec4 sensors[100]; 
  uniform int sensorCount;
  uniform float pH; 
  uniform float pV; 
  uniform vec3 clippingPlaneNormal;
  uniform float clippingPlaneConstant;

  void main() {
    vPosition = position;
    
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
    
    // Size attenuation with individual grain variation
    gl_PointSize = size * (15.0 / -mvPosition.z);
    
    vPosition = (modelMatrix * vec4(position, 1.0)).xyz;
    vViewZ = -mvPosition.z;
  }
`;

const fragmentShader = `
  varying float vTemperature;
  varying vec3 vPosition;
  varying float vViewZ;
  
  uniform float tMin;
  uniform float tMax;
  uniform vec3 clippingPlaneNormal;
  uniform float clippingPlaneConstant;

  // Google Turbo Color Palette (Approximation)
  vec3 turboColor(float x) {
    const vec4 kL = vec4(0.277977, -0.00570773, -0.0152427, -0.0181585);
    const vec4 kM = vec4(0.00448135, 2.37326, 0.449767, -0.0543594);
    const vec4 kN = vec4(0.485108, -2.60742, -0.320478, -0.191651);
    const vec4 kO = vec4(-0.0210287, 1.03158, 0.00406692, 0.000574045);
    const vec4 kP = vec4(1.0, 1.0, 1.0, 1.0);
    
    vec4 v = vec4(x, x * x, x * x * x, x * x * x * x);
    return vec3(dot(v, kL) + kO.x, dot(v, kM) + kO.y, dot(v, kN) + kO.z);
  }

  // Custom Heatmap Palette: Red -> Orange -> Yellow -> Green -> Blue
  vec3 heatmapColor(float t) {
    float v = clamp((t - tMin) / (tMax - tMin), 0.0, 1.0);
    
    // Vivid & Solid Colors for White Background
    vec3 c1 = vec3(1.0, 0.0, 0.0); // Red (Hot)
    vec3 c2 = vec3(1.0, 0.4, 0.0); // Orange
    vec3 c3 = vec3(1.0, 0.9, 0.0); // Yellow
    vec3 c4 = vec3(0.0, 0.8, 0.2); // Green
    vec3 c5 = vec3(0.0, 0.2, 1.0); // Blue (Cold)
    
    if (v > 0.75) return mix(c2, c1, (v - 0.75) * 4.0);
    if (v > 0.5) return mix(c3, c2, (v - 0.5) * 4.0);
    if (v > 0.25) return mix(c4, c3, (v - 0.25) * 4.0);
    return mix(c5, c4, v * 4.0);
  }

  void main() {
    // Clipping
    if (dot(vPosition, clippingPlaneNormal) > clippingPlaneConstant) {
        discard;
    }

    // Circular point shape with Gaussian-like falloff
    float r = distance(gl_PointCoord, vec2(0.5));
    if (r > 0.5) discard;
    
    float mask = (0.5 - r) * 2.0;
    mask = pow(mask, 0.3); // Sharp edges for less blur

    // Vivid Opacity mapping
    float tNorm = clamp((vTemperature - tMin) / (tMax - tMin), 0.0, 1.0);
    float alpha = mix(0.5, 0.95, pow(tNorm, 0.5)); 
    
    gl_FragColor = vec4(heatmapColor(vTemperature), alpha * mask);
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
  sensorLabels: CSS2DObject[] = []
  labelRenderer: CSS2DRenderer;
  hotLabel: CSS2DObject;
  coldLabel: CSS2DObject;
  clippingPlane: THREE.Plane;
  gui: GUI = new GUI();

  params = {
    slicing: 5,
    showSilo: true,
    showSensors: true,
    smoothnessH: 2.0,
    smoothnessV: 2.0,
    animateSensors: false,
    showAllTemperatures: true,
    autoRotate: true,
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
    this.scene.background = new THREE.Color(0xffffff);

    this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    this.camera.position.set(15, 15, 15);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.localClippingEnabled = true;
    document.getElementById('app')?.appendChild(this.renderer.domElement);

    this.labelRenderer = new CSS2DRenderer();
    this.labelRenderer.setSize(window.innerWidth, window.innerHeight);
    this.labelRenderer.domElement.style.position = 'absolute';
    this.labelRenderer.domElement.style.top = '0px';
    this.labelRenderer.domElement.style.pointerEvents = 'none';
    document.getElementById('app')?.appendChild(this.labelRenderer.domElement);

    this.hotLabel = this.createLabel('ZONA QUENTE', 'hot-label');
    this.coldLabel = this.createLabel('ZONA FRIA', 'cold-label');
    this.scene.add(this.hotLabel);
    this.scene.add(this.coldLabel);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.enablePan = true;
    this.controls.autoRotate = this.params.autoRotate;
    this.controls.autoRotateSpeed = 1.0;
    this.controls.maxPolarAngle = Math.PI / 2; // Limit rotation to 90 degrees (prevents looking from below)

    this.clippingPlane = new THREE.Plane(new THREE.Vector3(1, 0, 0), 5);

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
    this.sensorLabels.forEach(l => this.scene.remove(l));
    this.sensorLabels = [];

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

    // 3D Cylinder for cable
    const cableRadius = 0.03;
    const cableGeo = new THREE.CylinderGeometry(cableRadius, cableRadius, cableTop, 8);
    const cableMat = new THREE.MeshStandardMaterial({
      color: 0x444444,
      metalness: 0.8,
      roughness: 0.4,
      transparent: true,
      opacity: 0.8
    });
    const cableMesh = new THREE.Mesh(cableGeo, cableMat);
    cableMesh.position.set(x, cableTop / 2, z);
    cableMesh.name = 'cable';
    this.scene.add(cableMesh);

    for (let s = 0; s < count; s++) {
      const y = bOffset + (s / (count - 1)) * (sensorsTop - bOffset);
      const pos = new THREE.Vector3(x, y, z);

      // Use provided temp or initial random
      const temp = (temperatures && temperatures[s] !== undefined)
        ? temperatures[s]
        : TEMPERATURE_MIN + Math.random() * (TEMPERATURE_MAX - TEMPERATURE_MIN);

      this.sensors.push({ pos, temp, baseTemp: temp });

      const label = this.createLabel(`S${s + 1}`, 'sensor-label');
      label.position.copy(pos);
      this.scene.add(label);
      this.sensorLabels.push(label);

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

    // Metallic Texture (Procedural fallback or loaded image)
    const siloTexture = this.createCorrugatedTexture();
    siloTexture.wrapS = THREE.RepeatWrapping;
    siloTexture.wrapT = THREE.RepeatWrapping;
    siloTexture.repeat.set(12, 1); // Tile vertical ridges

    // Cylinder
    const cylinderGeo = new THREE.CylinderGeometry(SILO_RADIUS, SILO_RADIUS, SILO_HEIGHT, 128, 1, true);
    const material = new THREE.MeshStandardMaterial({
      color: 0xcccccc,
      map: siloTexture,
      side: THREE.BackSide,
      transparent: true,
      opacity: 0.25,
      metalness: 0.4,
      roughness: 0.7,
      clippingPlanes: [this.clippingPlane]
    });
    const cylinder = new THREE.Mesh(cylinderGeo, material);
    cylinder.position.y = SILO_HEIGHT / 2;
    group.add(cylinder);

    // Roof (Cone)
    const coneGeo = new THREE.ConeGeometry(SILO_RADIUS * 1.05, 3, 64, 1, true);
    const coneMaterial = new THREE.MeshStandardMaterial({
      color: 0xdddddd,
      map: siloTexture,
      transparent: true,
      opacity: 0.4,
      metalness: 0.3,
      roughness: 0.8,
      clippingPlanes: [this.clippingPlane]
    });
    const cone = new THREE.Mesh(coneGeo, coneMaterial);
    cone.position.y = SILO_HEIGHT + 1.5;
    group.add(cone);

    const floorGeo = new THREE.CircleGeometry(SILO_RADIUS * 1.5, 64);
    const floorMaterial = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      side: THREE.DoubleSide
    });
    const floor = new THREE.Mesh(floorGeo, floorMaterial);
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -0.01;
    group.add(floor);

    // Subtle Ground Shadow (Circular Gradient)
    const shadowGeo = new THREE.CircleGeometry(SILO_RADIUS * 1.5, 64);
    const shadowCanvas = document.createElement('canvas');
    shadowCanvas.width = 128;
    shadowCanvas.height = 128;
    const ctx = shadowCanvas.getContext('2d')!;
    const grad = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
    grad.addColorStop(0, 'rgba(0,0,0,0.2)');
    grad.addColorStop(0.8, 'rgba(0,0,0,0.05)');
    grad.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 128, 128);
    const shadowTex = new THREE.CanvasTexture(shadowCanvas);
    const shadowMat = new THREE.MeshBasicMaterial({ map: shadowTex, transparent: true, depthWrite: false });
    const shadow = new THREE.Mesh(shadowGeo, shadowMat);
    shadow.rotation.x = -Math.PI / 2;
    shadow.position.y = 0.01;
    this.scene.add(shadow);

    this.scene.add(group);
  }

  createCorrugatedTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d')!;

    // Base metallic color
    ctx.fillStyle = '#999999';
    ctx.fillRect(0, 0, 256, 256);

    // Draw vertical corrugated ridges
    for (let x = 0; x < 256; x += 32) {
      const gradient = ctx.createLinearGradient(x, 0, x + 32, 0);
      gradient.addColorStop(0, 'rgba(0,0,0,0.15)');
      gradient.addColorStop(0.5, 'rgba(255,255,255,0.15)');
      gradient.addColorStop(1, 'rgba(0,0,0,0.15)');
      ctx.fillStyle = gradient;
      ctx.fillRect(x, 0, 32, 256);
    }

    // Add some noise for realism
    for (let i = 0; i < 5000; i++) {
      const x = Math.random() * 256;
      const y = Math.random() * 256;
      const val = Math.random() * 20;
      ctx.fillStyle = `rgba(${val},${val},${val},0.05)`;
      ctx.fillRect(x, y, 1, 1);
    }

    const texture = new THREE.CanvasTexture(canvas);
    texture.anisotropy = 16;
    return texture;
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
    const count = 60000;
    const positions = new Float32Array(count * 3);
    const sizes = new Float32Array(count); // Individual sizes

    for (let i = 0; i < count; i++) {
      const r = Math.sqrt(Math.random()) * SILO_RADIUS;
      const theta = Math.random() * Math.PI * 2;
      const x = r * Math.cos(theta);
      const z = r * Math.sin(theta);

      // Interpolate base height from cables
      let weightSum = 0;
      let heightSum = 0;
      for (const cl of this.cableLevels) {
        const dist = cl.pos.distanceTo(new THREE.Vector2(x, z));
        const weight = 1.0 / (Math.pow(dist, 2.0) + 0.01);
        weightSum += weight;
        heightSum += cl.height * weight;
      }

      let yMax = weightSum > 0 ? (heightSum / weightSum) : GRAIN_HEIGHT;

      // Pile Effect (Natural Heap/Angle of Repose)
      const distFromCenter = r / SILO_RADIUS;
      const heapHeight = 1.8;
      const domeEffect = Math.cos(distFromCenter * (Math.PI / 2)) * heapHeight;
      yMax += domeEffect;

      const noise = (Math.random() - 0.5) * 0.2;
      yMax += noise;

      const y = Math.random() * yMax;

      positions[i * 3] = x;
      positions[i * 3 + 1] = y;
      positions[i * 3 + 2] = z;

      // Random grain size between 14 and 26
      sizes[i] = 14 + Math.random() * 12;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

    const sensorData = new Float32Array(100 * 4);
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
      transparent: true,
      depthWrite: false,
      blending: THREE.NormalBlending
    });

    this.grainPoints = new THREE.Points(geometry, material);
    this.grainPoints.name = 'grain-mass';
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
      this.sensorLabels.forEach(l => l.visible = v && this.params.showAllTemperatures);
    });

    folder.add(this.params, 'showAllTemperatures').name('Mostrar Temperaturas').onChange((v: boolean) => {
      this.sensorLabels.forEach(l => l.visible = v && this.params.showSensors);
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
    this.labelRenderer.render(this.scene, this.camera);
    this.updateExtremeLabels();
    this.updateAllSensorLabels();
  }

  createLabel(text: string, className: string): CSS2DObject {
    const div = document.createElement('div');
    div.className = `label-3d ${className}`;

    const content = document.createElement('div');
    content.className = 'label-content';
    content.innerHTML = `<strong>${text}</strong><br><span class="temp-val">--.-°C</span>`;
    div.appendChild(content);

    const label = new CSS2DObject(div);
    label.position.set(0, 0, 0);
    return label;
  }

  updateExtremeLabels() {
    if (this.sensors.length === 0) return;

    let hot = this.sensors[0];
    let cold = this.sensors[0];

    this.sensors.forEach(s => {
      if (s.temp > hot.temp) hot = s;
      if (s.temp < cold.temp) cold = s;
    });

    this.hotLabel.position.copy(hot.pos);
    this.hotLabel.element.querySelector('.temp-val')!.textContent = `${hot.temp.toFixed(1)}°C`;

    this.coldLabel.position.copy(cold.pos);
    this.coldLabel.element.querySelector('.temp-val')!.textContent = `${cold.temp.toFixed(1)}°C`;

    // Visibility based on clipping
    const isHotVisible = hot.pos.x <= this.clippingPlane.constant;
    const isColdVisible = cold.pos.x <= this.clippingPlane.constant;

    this.hotLabel.visible = this.params.showSensors && isHotVisible;
    this.coldLabel.visible = this.params.showSensors && isColdVisible;
  }

  updateAllSensorLabels() {
    this.sensors.forEach((s, i) => {
      const label = this.sensorLabels[i];
      if (label) {
        const isVisible = s.pos.x <= this.clippingPlane.constant;
        label.visible = this.params.showSensors && this.params.showAllTemperatures && isVisible;
        if (label.visible) {
          const tempVal = label.element.querySelector('.temp-val')!;
          tempVal.textContent = `${s.temp.toFixed(1)}°C`;

          // Dynamic Coloring
          const color = this.getTemperatureColor(s.temp);
          const content = label.element.querySelector('.label-content') as HTMLElement;
          if (content) {
            content.style.backgroundColor = color;
            label.element.style.color = color; // For the ::after arrow color via border-color: inherit
          }
        }
      }
    });
  }

  getTemperatureColor(t: number): string {
    const v = Math.max(0, Math.min(1, (t - TEMPERATURE_MIN) / (TEMPERATURE_MAX - TEMPERATURE_MIN)));

    const c1 = new THREE.Color(1.0, 0.0, 0.0); // Red
    const c2 = new THREE.Color(1.0, 0.4, 0.0); // Orange
    const c3 = new THREE.Color(1.0, 0.9, 0.0); // Yellow
    const c4 = new THREE.Color(0.0, 0.8, 0.2); // Green
    const c5 = new THREE.Color(0.0, 0.2, 1.0); // Blue

    let finalColor = new THREE.Color();
    if (v > 0.75) finalColor.lerpColors(c2, c1, (v - 0.75) * 4.0);
    else if (v > 0.5) finalColor.lerpColors(c3, c2, (v - 0.5) * 4.0);
    else if (v > 0.25) finalColor.lerpColors(c4, c3, (v - 0.25) * 4.0);
    else finalColor.lerpColors(c5, c4, v * 4.0);

    return `#${finalColor.getHexString()}`;
  }
}

new SiloApp();
