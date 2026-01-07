"use client"

import { useState } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { ArrowLeft, Upload } from "lucide-react"
import Image from "next/image"

export default function ModelPage() {
  const [file, setFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<number | null>(null)

  const handleFile = async (selectedFile: File) => {
    setFile(selectedFile)
    setLoading(true)
    setResult(null)

    const formData = new FormData()
    formData.append("file", selectedFile)

    try {
      // ✅ CHANGE 1: use backend URL from env variable
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/predict`,
        {
          method: "POST",
          body: formData,
        }
      )

      // ✅ CHANGE 2: safety check
      if (!response.ok) {
        throw new Error("Server error")
      }

      const data = await response.json()

      if (data.error) {
        alert(data.error)
        setLoading(false)
        setFile(null)
        return
      }

      const percentage = data.probability * 100
      setResult(percentage)
      setLoading(false)
    } catch (err) {
      console.error("Prediction error:", err)
      alert("Prediction failed")
      setLoading(false)
      setFile(null)
    }
  }

  const resetUpload = () => {
    setFile(null)
    setResult(null)
    setLoading(false)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-cyan-50 via-blue-50 to-teal-50">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-white/70 backdrop-blur-md border-b border-cyan-200">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <Link href="/" className="flex items-center">
            <Image
              src="/images/upscalemedia-transformed.png"
              alt="OncoMap Logo"
              width={50}
              height={50}
              className="rounded-full"
            />
          </Link>
          <div className="flex items-center gap-6">
            <Link href="/" className="text-sm font-medium text-gray-700 hover:text-cyan-600 transition-colors">
              Home
            </Link>
            <Link href="/model" className="text-sm font-medium text-gray-700 hover:text-cyan-600 transition-colors">
              Our Model
            </Link>
            <Link href="/business" className="text-sm font-medium text-gray-700 hover:text-cyan-600 transition-colors">
              Business
            </Link>
          </div>
        </div>
      </nav>

      <div className="pt-24 pb-16">
        {/* Back Button */}
        <div className="container mx-auto px-4 mb-8">
          <Button variant="ghost" asChild className="text-cyan-700 hover:text-cyan-800 hover:bg-cyan-100">
            <Link href="/">
              <ArrowLeft className="mr-2 h-4 w-4" /> Back to Home
            </Link>
          </Button>
        </div>

        {/* Background Section */}
        <section className="container mx-auto px-4 mb-24">
          <h1 className="text-4xl md:text-5xl font-bold mb-8 text-center bg-gradient-to-r from-cyan-600 to-teal-600 bg-clip-text text-transparent">
            Our Predictive Models
          </h1>
          <div className="max-w-4xl mx-auto">
            <div className="bg-gradient-to-br from-white/80 to-cyan-50/50 border border-cyan-200 rounded-lg p-8 mb-12 shadow-sm">
              <h2 className="text-2xl font-semibold mb-4 text-cyan-700">Background</h2>
              <p className="text-lg text-gray-700 leading-relaxed mb-4">
                Cancer immunotherapy has revolutionized oncology, but predicting which patients will respond remains a
                significant challenge. Our models leverage tumor transcriptomic signatures to predict treatment response
                with unprecedented accuracy.
              </p>
              <p className="text-lg text-gray-700 leading-relaxed">
                By analyzing gene expression patterns from thousands of tumor samples, we've developed machine learning
                models that can identify patients most likely to benefit from specific immunotherapy treatments,
                reducing unnecessary side effects and healthcare costs while improving outcomes.
              </p>
            </div>
          </div>
        </section>

        {/* Immunotherapy Prediction Tool */}
        <section className="container mx-auto px-4 mb-24">
          <div className="max-w-2xl mx-auto">
            <h2 className="text-3xl font-bold mb-8 text-center bg-gradient-to-r from-cyan-600 to-blue-600 bg-clip-text text-transparent">
              Head & Neck Cancer Immunotherapy Prediction
            </h2>

            <div className="bg-white/80 border border-cyan-200 rounded-lg p-8 shadow-sm">
              <p className="text-center text-gray-700 mb-8">
                Upload patient transcriptomic data for real-time immunotherapy response prediction
              </p>

              {!loading && result === null && (
                <div
                  className="border-2 border-dashed border-cyan-300 rounded-lg p-12 text-center cursor-pointer hover:border-cyan-400 hover:bg-cyan-50 transition-colors"
                  onClick={() => document.getElementById("fileInput")?.click()}
                  onDrop={(e) => {
                    e.preventDefault()
                    if (e.dataTransfer.files.length > 0) {
                      handleFile(e.dataTransfer.files[0])
                    }
                  }}
                >
                  <Upload className="h-8 w-8 text-cyan-600 mx-auto mb-4" />
                  <p className="font-semibold mb-2">Drop file here or click to upload</p>
                </div>
              )}

              {loading && <p className="text-center">Analyzing patient data…</p>}

              {result !== null && (
                <div className="text-center py-12">
                  <h3 className="text-xl font-semibold mb-6">Immunotherapy Success Probability</h3>
                  <div className="text-7xl font-bold">{result.toFixed(1)}%</div>
                  <Button onClick={resetUpload} className="mt-6">
                    Upload Another File
                  </Button>
                </div>
              )}

              <input
                type="file"
                id="fileInput"
                className="hidden"
                accept=".csv,.txt,.json"
                onChange={(e) => {
                  if (e.target.files && e.target.files.length > 0) {
                    handleFile(e.target.files[0])
                  }
                }}
              />
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}
