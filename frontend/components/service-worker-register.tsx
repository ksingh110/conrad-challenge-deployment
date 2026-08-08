"use client";

import { useEffect } from "react";

export default function ServiceWorkerRegister() {
  useEffect(() => {
    if ("serviceWorker" in navigator) {
      navigator.serviceWorker
        .register("/sw.js")
        .then(() => {
          console.log("OncoMap service worker registered");
        })
        .catch((error) => {
          console.error("Service worker registration failed:", error);
        });
    }
  }, []);

  return null;
}