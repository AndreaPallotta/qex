"""
Bloch sphere visualization: density matrix → (x,y,z) coordinates → HTML artifact.
"""

from typing import Tuple
import numpy as np


def density_matrix_to_bloch(rho: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert a 1-qubit density matrix to Bloch sphere coordinates (x, y, z).
    
    For a density matrix ρ, the Bloch vector is computed as:
        x = Tr(ρ · σ_x)
        y = Tr(ρ · σ_y)
        z = Tr(ρ · σ_z)
    
    where σ_x, σ_y, σ_z are the Pauli matrices.
    
    Args:
        rho: 2x2 density matrix (complex dtype).
            Must be a valid density matrix (Hermitian, trace=1, positive semidefinite).
    
    Returns:
        Tuple of (x, y, z) coordinates as real floats.
        Coordinates are guaranteed to satisfy x² + y² + z² ≤ 1.
    """
    # Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Compute Bloch coordinates: Tr(ρ · σ_i)
    x = np.real(np.trace(rho @ sigma_x))
    y = np.real(np.trace(rho @ sigma_y))
    z = np.real(np.trace(rho @ sigma_z))
    
    return (float(x), float(y), float(z))


def bloch_to_html(x: float, y: float, z: float, title: str = "Bloch Sphere") -> str:
    """
    Generate an HTML artifact visualizing a point on the Bloch sphere.
    
    The HTML should:
    - Render a 3D Bloch sphere
    - Mark the point (x, y, z) on the sphere
    - Display the coordinates
    - Be self-contained (no external dependencies, or use CDN for 3D library)
    - Be viewable in a browser
    
    Args:
        x: X coordinate on Bloch sphere.
        y: Y coordinate on Bloch sphere.
        z: Z coordinate on Bloch sphere.
        title: Optional title for the visualization.
    
    Returns:
        Complete HTML string that can be saved to a file and opened in a browser.
    """
    # Use Three.js via CDN for 3D visualization
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
        body {{
            margin: 0;
            font-family: Arial, sans-serif;
            background: #1a1a1a;
            color: #fff;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }}
        #container {{
            width: 800px;
            height: 600px;
            border: 2px solid #444;
            border-radius: 8px;
            margin: 20px 0;
        }}
        #info {{
            text-align: center;
            margin: 10px 0;
        }}
        .coord {{
            display: inline-block;
            margin: 0 15px;
            font-family: monospace;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div id="container"></div>
    <div id="info">
        <div class="coord">x = {x:.4f}</div>
        <div class="coord">y = {y:.4f}</div>
        <div class="coord">z = {z:.4f}</div>
    </div>
    <script>
        // Scene setup
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, 800/600, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(800, 600);
        document.getElementById('container').appendChild(renderer.domElement);
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5, 5, 5);
        scene.add(directionalLight);
        
        // Bloch sphere (unit sphere)
        const sphereGeometry = new THREE.SphereGeometry(1, 32, 32);
        const sphereMaterial = new THREE.MeshPhongMaterial({{
            color: 0x4a90e2,
            transparent: true,
            opacity: 0.3,
            side: THREE.DoubleSide,
            wireframe: false
        }});
        const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
        scene.add(sphere);
        
        // Wireframe overlay
        const wireframe = new THREE.WireframeGeometry(sphereGeometry);
        const wireframeLine = new THREE.LineSegments(wireframe, new THREE.LineBasicMaterial({{ color: 0xffffff, opacity: 0.2 }}));
        scene.add(wireframeLine);
        
        // Axes
        const axesHelper = new THREE.AxesHelper(1.2);
        scene.add(axesHelper);
        
        // Point marker
        const pointGeometry = new THREE.SphereGeometry(0.05, 16, 16);
        const pointMaterial = new THREE.MeshPhongMaterial({{ color: 0xff4444 }});
        const point = new THREE.Mesh(pointGeometry, pointMaterial);
        point.position.set({x}, {y}, {z});
        scene.add(point);
        
        // Line from origin to point
        const lineGeometry = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(0, 0, 0),
            new THREE.Vector3({x}, {y}, {z})
        ]);
        const lineMaterial = new THREE.LineBasicMaterial({{ color: 0xff4444, linewidth: 2 }});
        const line = new THREE.Line(lineGeometry, lineMaterial);
        scene.add(line);
        
        // Camera position
        camera.position.set(2.5, 2.5, 2.5);
        camera.lookAt(0, 0, 0);
        
        // Animation loop
        let angle = 0;
        function animate() {{
            requestAnimationFrame(animate);
            angle += 0.01;
            camera.position.x = 2.5 * Math.cos(angle);
            camera.position.z = 2.5 * Math.sin(angle);
            camera.lookAt(0, 0, 0);
            renderer.render(scene, camera);
        }}
        animate();
    </script>
</body>
</html>"""
    return html
