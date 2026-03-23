---
title: "Projects"
permalink: /projects/
layout: single
author_profile: true
classes: wide
---

<style>
.projects-intro {
  margin-bottom: 1.5rem;
}

.projects-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 1.5rem;
  margin-top: 2rem;
}

.project-card {
  overflow: hidden;
  border: 1px solid rgba(127, 127, 127, 0.25);
  border-radius: 14px;
  background: rgba(127, 127, 127, 0.08);
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}

.project-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 10px 24px rgba(0, 0, 0, 0.14);
  border-color: rgba(127, 127, 127, 0.45);
}

.project-card img {
  display: block;
  width: 100%;
  height: 220px;
  object-fit: cover;
  border-bottom: 1px solid rgba(127, 127, 127, 0.2);
}

.project-card-body {
  padding: 1.25rem 1.5rem 1.5rem;
}

.project-card h2 {
  margin-top: 0;
  margin-bottom: 0.75rem;
  font-size: 1.25rem;
  line-height: 1.3;
}

.project-card p {
  margin-bottom: 1rem;
}

.project-card a {
  text-decoration: none;
}

.project-card a.project-title {
  color: inherit;
}

.project-card a.project-link {
  display: inline-block;
  font-weight: 600;
  border-bottom: 2px solid transparent;
}

.project-card a.project-link:hover {
  border-color: currentColor;
}
</style>

<div class="projects-intro">
  A selection of interactive tools and visual explorations from my portfolio.
</div>

<div class="projects-grid">

  <div class="project-card">
    <a href="/projects/network-graphing-explorer/">
      <img src="/images/network-graph-explorer-preview.png" alt="Preview of the Network Graph Explorer">
    </a>
    <div class="project-card-body">
      <h2>
        <a class="project-title" href="/projects/network-graphing-explorer/">
          Network Graph Explorer
        </a>
      </h2>
      <p>
        Explore graph structures, nodes, edges, and connectivity patterns through an interactive network visualization.
      </p>
      <a class="project-link" href="/projects/network-graphing-explorer/">
        Open project →
      </a>
    </div>
  </div>

  <div class="project-card">
    <a href="/projects/machine-learning-model-explorer/">
      <img src="/images/machine-learning-model-explorer-preview.png" alt="Preview of the Machine Learning Model Explorer">
    </a>
    <div class="project-card-body">
      <h2>
        <a class="project-title" href="/projects/machine-learning-model-explorer/">
          Machine Learning Model Explorer
        </a>
      </h2>
      <p>
        Interact with machine learning concepts, model outputs, and model behavior in a hands-on exploratory interface.
      </p>
      <a class="project-link" href="/projects/machine-learning-model-explorer/">
        Open project →
      </a>
    </div>
  </div>

  <div class = "project-card">
    <a href = "/projects/regression-model-explorer/">
      <img src = "/images/regressionpipe.png" alt = "Preview of the Forecasting Explorer">
    </a>
    <div class = "project-card-body">
      <h2>
        <a class = "project-title" href = "/projects/regression-model-explorer/">
          Forecasting and Regression Explorer
        </a>
      </h2>
      <p>
        Explore regression and forecasting techniques with generalized-additive models, and more.
      </p>
      <a class="project-link" href="/projects/regression-model-explorer/">
        Open project →
      </a>
    </div>
  </div>
</div>