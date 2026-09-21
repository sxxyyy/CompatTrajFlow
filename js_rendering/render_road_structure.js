#!/usr/bin/env node
/**
 * Render Road-Network-Structure view PNGs from trajectory JSON + road network GeoJSON.
 *
 * Adapted from Traj-MLLM's 3_vis_road_structure.js.
 *
 * Usage:
 *   node render_road_structure.js --json-dir <dir> --image-dir <dir>
 *        --nodes-geojson <file> --edges-geojson <file> [--html-dir <dir>]
 *
 * The nodes and edges GeoJSON files are generated once by the Python
 * preprocessing step from the road-network shapefiles.
 */

const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer');

function parseArgs() {
    const args = process.argv.slice(2);
    const opts = {
        jsonDir: process.env.TRAJ_MLLM_JSON_DIR || '../data/ano_trajectory_jsons',
        imageDir: process.env.TRAJ_MLLM_IMAGE_DIR || '../data/road_structure_anomaly_images',
        htmlDir: process.env.TRAJ_MLLM_HTML_DIR || '../data/road_structure_anomaly_htmls',
        nodesGeoJson: process.env.TRAJ_MLLM_NODES_GEOJSON || '',
        edgesGeoJson: process.env.TRAJ_MLLM_EDGES_GEOJSON || '',
        concurrency: parseInt(process.env.TRAJ_MLLM_CONCURRENCY || '4', 10),
    };
    for (let i = 0; i < args.length; i++) {
        switch (args[i]) {
            case '--json-dir': opts.jsonDir = args[++i]; break;
            case '--image-dir': opts.imageDir = args[++i]; break;
            case '--html-dir': opts.htmlDir = args[++i]; break;
            case '--nodes-geojson': opts.nodesGeoJson = args[++i]; break;
            case '--edges-geojson': opts.edgesGeoJson = args[++i]; break;
            case '--concurrency': case '-c': opts.concurrency = parseInt(args[++i], 10) || 4; break;
        }
    }
    return opts;
}

const opts = parseArgs();

[opts.imageDir, opts.htmlDir].forEach(d => {
    if (!fs.existsSync(d)) fs.mkdirSync(d, { recursive: true });
});

let nodesGeoJsonCache = null;
let edgesGeoJsonCache = null;

function loadRoadNetwork() {
    if (!nodesGeoJsonCache && opts.nodesGeoJson && fs.existsSync(opts.nodesGeoJson)) {
        nodesGeoJsonCache = JSON.parse(fs.readFileSync(opts.nodesGeoJson, 'utf8'));
        console.log('Loaded nodes:', nodesGeoJsonCache.features.length);
    }
    if (!edgesGeoJsonCache && opts.edgesGeoJson && fs.existsSync(opts.edgesGeoJson)) {
        edgesGeoJsonCache = JSON.parse(fs.readFileSync(opts.edgesGeoJson, 'utf8'));
        console.log('Loaded edges:', edgesGeoJsonCache.features.length);
    }
}

function calculateExtendedBounds(trajectoryData, buffer) {
    let minLat = Infinity, maxLat = -Infinity, minLng = Infinity, maxLng = -Infinity;
    for (const point of trajectoryData) {
        minLng = Math.min(minLng, point[0]);
        maxLng = Math.max(maxLng, point[0]);
        minLat = Math.min(minLat, point[1]);
        maxLat = Math.max(maxLat, point[1]);
    }
    return { minLng: minLng - buffer, maxLng: maxLng + buffer, minLat: minLat - buffer, maxLat: maxLat + buffer };
}

function isPointInBounds(point, bounds) {
    return point[0] >= bounds.minLng && point[0] <= bounds.maxLng &&
        point[1] >= bounds.minLat && point[1] <= bounds.maxLat;
}

function isLineInBounds(line, bounds) {
    for (const p of line) { if (isPointInBounds(p, bounds)) return true; }
    return false;
}

function distanceToSegment(p, v, w) {
    const l2 = Math.pow(v[0] - w[0], 2) + Math.pow(v[1] - w[1], 2);
    if (l2 === 0) return Math.sqrt(Math.pow(p[0] - v[0], 2) + Math.pow(p[1] - v[1], 2));
    let t = ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2;
    t = Math.max(0, Math.min(1, t));
    return Math.sqrt(Math.pow(p[0] - (v[0] + t * (w[0] - v[0])), 2) + Math.pow(p[1] - (v[1] + t * (w[1] - v[1])), 2));
}

function minDistanceToTrajectory(point, trajectoryData) {
    let minDist = Infinity;
    for (let i = 0; i < trajectoryData.length - 1; i++) {
        minDist = Math.min(minDist, distanceToSegment(
            [point[1], point[0]],
            [trajectoryData[i][1], trajectoryData[i][0]],
            [trajectoryData[i + 1][1], trajectoryData[i + 1][0]]
        ));
    }
    return minDist;
}

function minDistanceFromTrajectoryToSegment(trajectoryData, segment) {
    let minDist = Infinity;
    for (const point of trajectoryData) {
        minDist = Math.min(minDist, distanceToSegment(
            [point[1], point[0]],
            [segment[0][1], segment[0][0]],
            [segment[1][1], segment[1][0]]
        ));
    }
    return minDist;
}

const DIST_THRESHOLD_DEG = 0.001;  // ~111 m

function preFilterNodes(trajectoryData, bounds) {
    if (!nodesGeoJsonCache) return { type: 'FeatureCollection', features: [] };
    const filtered = nodesGeoJsonCache.features.filter(f => {
        const coords = f.geometry.coordinates;
        return isPointInBounds(coords, bounds) && minDistanceToTrajectory(coords, trajectoryData) < DIST_THRESHOLD_DEG * 111000;
    });
    return { type: 'FeatureCollection', features: filtered };
}

function preFilterEdges(trajectoryData, bounds) {
    if (!edgesGeoJsonCache) return { type: 'FeatureCollection', features: [] };
    const filtered = edgesGeoJsonCache.features.filter(f => {
        const coords = f.geometry.coordinates;
        if (!isLineInBounds(coords, bounds)) return false;
        for (let i = 0; i < coords.length - 1; i++) {
            if (minDistanceFromTrajectoryToSegment(trajectoryData, [coords[i], coords[i + 1]]) < DIST_THRESHOLD_DEG * 111000) return true;
        }
        return false;
    });
    return { type: 'FeatureCollection', features: filtered };
}

function calculateTrajectoryRange(trajectoryData) {
    if (!trajectoryData || trajectoryData.length < 2) return { radius: 4, width: 1.2 };
    let minLat = Infinity, maxLat = -Infinity, minLng = Infinity, maxLng = -Infinity;
    for (const p of trajectoryData) {
        minLng = Math.min(minLng, p[0]); maxLng = Math.max(maxLng, p[0]);
        minLat = Math.min(minLat, p[1]); maxLat = Math.max(maxLat, p[1]);
    }
    const diagonalDistance = Math.sqrt(
        Math.pow((maxLng - minLng) * 111320 * Math.cos((minLat + maxLat) / 2 * Math.PI / 180), 2) +
        Math.pow((maxLat - minLat) * 110540, 2)
    );
    if (diagonalDistance <= 500) return { radius: 6, width: 2.5 };
    else if (diagonalDistance <= 1000) return { radius: 5.8, width: 2 };
    else if (diagonalDistance <= 1500) return { radius: 5, width: 1.8 };
    else if (diagonalDistance <= 2000) return { radius: 4.5, width: 1.5 };
    else return { radius: 4, width: 1.2 };
}

const createHtmlTemplate = (trajectory) => {
    const lineArr = trajectory.o_geo || [];
    const trajectoryPoints = JSON.stringify(lineArr);
    const trajectoryBounds = calculateExtendedBounds(lineArr, 0.01);
    const filteredNodes = preFilterNodes(lineArr, trajectoryBounds);
    const filteredEdges = preFilterEdges(lineArr, trajectoryBounds);

    // Compute adaptive styling in Node.js and embed as literals.
    const style = calculateTrajectoryRange(lineArr);

    return `<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="initial-scale=1.0, user-scalable=no, width=device-width">
    <title>Road Network - ${trajectory.devid || 'Unknown'}</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        html, body, #map { height: 100%; width: 100%; margin: 0; padding: 0; }
    </style>
</head>
<body>
    <div id="map"></div>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        const nodesGeoJson = ${JSON.stringify(filteredNodes)};
        const edgesGeoJson = ${JSON.stringify(filteredEdges)};

        // Initial view as fallback; fitBounds will override.
        const map = L.map('map').setView([41.15, -8.61], 13);
        L.tileLayer('http://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png', {
            attribution: 'Positron'
        }).addTo(map);

        const trajectoryData = ${trajectoryPoints};
        const fullPathLine = L.polyline([], { color: 'Lime', weight: 10, opacity: 0.6 }).addTo(map);

        const startIcon = L.divIcon({
            html: '<div style="width:30px;height:30px;border-radius:50%;background:#4CAF50;color:white;display:flex;justify-content:center;align-items:center;font-weight:bold;font-size:14px;">S</div>',
            className: 'custom-start-icon', iconSize: [10, 10], iconAnchor: [5, 5]
        });
        const endIcon = L.divIcon({
            html: '<div style="width:30px;height:30px;border-radius:50%;background:#F44336;color:white;display:flex;justify-content:center;align-items:center;font-weight:bold;font-size:14px;">E</div>',
            className: 'custom-end-icon', iconSize: [10, 10], iconAnchor: [5, 5]
        });

        const points = trajectoryData.map(p => [p[1], p[0]]);
        fullPathLine.setLatLngs(points);

        // Display road network — adaptive styling computed at template-gen time.
        L.geoJSON(nodesGeoJson, {
            pointToLayer: function(feature, latlng) {
                return L.circleMarker(latlng, {
                    radius: ${style.radius}, fillColor: 'red', color: 'darkred',
                    weight: 1, opacity: 0.8, fillOpacity: 0.6
                });
            }
        }).addTo(map);
        L.geoJSON(edgesGeoJson, {
            style: function() { return {
                color: 'blue', weight: ${style.width}, opacity: 0.7
            }; }
        }).addTo(map);

        // Fit without animation so the view is settled before screenshot.
        map.fitBounds(fullPathLine.getBounds(), { padding: [60, 60], animate: false });
        map.whenReady(function() {
            L.marker(points[0], {icon: startIcon}).addTo(map);
            L.marker(points[points.length - 1], {icon: endIcon}).addTo(map);
        });
    </script>
</body>
</html>`;
};

async function renderAndCaptureTrajectory(browser, trajectoryFilePath) {
    try {
        const trajectoryData = JSON.parse(fs.readFileSync(trajectoryFilePath, 'utf8'));
        const deviceId = path.basename(trajectoryFilePath, '.json');

        const imagePath = path.join(opts.imageDir, `${deviceId}.png`);
        if (fs.existsSync(imagePath)) {
            return { imagePath, skipped: true };
        }

        const htmlContent = createHtmlTemplate(trajectoryData);
        const htmlFilePath = path.join(opts.htmlDir, `${deviceId}.html`);
        fs.writeFileSync(htmlFilePath, htmlContent);

        const page = await browser.newPage();
        try {
            await page.setViewport({ width: 1920, height: 1920 });

            await page.goto(`file://${path.resolve(htmlFilePath)}`, {
                waitUntil: 'networkidle2', timeout: 60000
            });

            await new Promise(resolve => setTimeout(resolve, 2000));

            const clipRect = await page.evaluate(() => {
                const bounds = fullPathLine.getBounds();
                if (!bounds || !bounds.isValid()) return null;
                const nw = map.latLngToContainerPoint(bounds.getNorthWest());
                const se = map.latLngToContainerPoint(bounds.getSouthEast());
                const pad = 80;
                return {
                    x: Math.max(0, nw.x - pad), y: Math.max(0, nw.y - pad),
                    width: Math.min(1920, se.x - nw.x + 2 * pad),
                    height: Math.min(1920, se.y - nw.y + 2 * pad)
                };
            });
            const screenshotOpts = clipRect
                ? { path: imagePath, clip: clipRect }
                : { path: imagePath };
            await page.screenshot(screenshotOpts);
        } finally {
            await page.close();
        }
        return { imagePath };
    } catch (error) {
        console.error('Error rendering:', error);
        return null;
    }
}

async function main() {
    loadRoadNetwork();
    if (!fs.existsSync(opts.jsonDir)) {
        console.error('JSON directory not found:', opts.jsonDir);
        process.exit(1);
    }
    const files = fs.readdirSync(opts.jsonDir)
        .filter(f => f.endsWith('.json'))
        .sort((a, b) => (parseInt(a, 10) || 0) - (parseInt(b, 10) || 0));
    if (files.length === 0) { console.log('No trajectory JSON files found.'); return; }
    console.log(`Found ${files.length} trajectory files (global + segments).`);
    console.log(`Concurrency: ${opts.concurrency} pages`);

    const browser = await puppeteer.launch({
        executablePath: process.env.PUPPETEER_EXECUTABLE_PATH || '/usr/bin/google-chrome',
        args: [
            '--allow-file-access-from-files', '--disable-web-security',
            '--disable-features=IsolateOrigins,site-per-process',
            '--no-sandbox', '--disable-setuid-sandbox',
        ]
    });

    try {
        let rendered = 0;
        let skipped = 0;
        let failed = 0;
        for (let i = 0; i < files.length; i += opts.concurrency) {
            const chunk = files.slice(i, i + opts.concurrency);
            const tasks = chunk.map((file) => {
                const filePath = path.join(opts.jsonDir, file);
                return renderAndCaptureTrajectory(browser, filePath);
            });
            const results = await Promise.all(tasks);
            for (const result of results) {
                if (result) {
                    if (result.skipped) { skipped++; } else { rendered++; }
                } else {
                    failed++;
                }
            }
            const done = Math.min(i + opts.concurrency, files.length);
            console.log(`Progress: ${done}/${files.length} (${rendered} new, ${skipped} cached, ${failed} failed)`);
        }
        console.log(`Done. Rendered: ${rendered}, Skipped: ${skipped}, Failed: ${failed}`);
    } finally {
        await browser.close();
    }
}

main().catch(err => { console.error(err); process.exit(1); });
