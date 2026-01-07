import Link from "next/link"
import { Button } from "@/components/ui/button"
import { ArrowLeft, TrendingUp, Users, Building2, Target } from "lucide-react"
import Image from "next/image"

export default function BusinessPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-cyan-50 via-blue-50 to-teal-50">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-white/70 backdrop-blur-md border-b border-cyan-200">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          {/* Logo only */}
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

        {/* Hero Section */}
        <section className="container mx-auto px-4 mb-24">
          <div className="max-w-4xl mx-auto text-center">
            <h1 className="text-4xl md:text-5xl font-bold mb-6 bg-gradient-to-r from-cyan-600 to-teal-600 bg-clip-text text-transparent">
              Business Overview
            </h1>
            <p className="text-xl text-gray-700 leading-relaxed">
              Transforming precision oncology into a scalable, sustainable business model that improves patient outcomes
              while reducing healthcare costs.
            </p>
          </div>
        </section>

        {/* Market Opportunity */}
        <section className="container mx-auto px-4 mb-24">
          <div className="max-w-6xl mx-auto">
            <h2 className="text-3xl font-bold mb-12 text-center bg-gradient-to-r from-teal-600 to-cyan-600 bg-clip-text text-transparent">
              Market Opportunity
            </h2>
            <div className="grid md:grid-cols-3 gap-8">
              {/* Light gradient cards */}
              <div className="bg-gradient-to-br from-cyan-100/40 to-blue-100/30 border border-cyan-200 rounded-lg p-6 shadow-sm">
                <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-500 flex items-center justify-center mb-4">
                  <TrendingUp className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-xl font-semibold mb-3 text-cyan-700">Growing Market</h3>
                <p className="text-gray-700 leading-relaxed">
                  The global cancer immunotherapy market is projected to reach $200B+ by 2028, with precision oncology
                  driving significant growth.
                </p>
              </div>

              <div className="bg-gradient-to-br from-teal-100/40 to-cyan-100/30 border border-cyan-200 rounded-lg p-6 shadow-sm">
                <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-teal-500 to-cyan-500 flex items-center justify-center mb-4">
                  <Users className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-xl font-semibold mb-3 text-teal-700">Unmet Need</h3>
                <p className="text-gray-700 leading-relaxed">
                  Only 20-30% of patients respond to immunotherapy, creating urgent demand for predictive tools that
                  identify optimal candidates.
                </p>
              </div>

              <div className="bg-gradient-to-br from-blue-100/30 to-teal-100/40 border border-cyan-200 rounded-lg p-6 shadow-sm">
                <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-blue-500 to-teal-500 flex items-center justify-center mb-4">
                  <Target className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-xl font-semibold mb-3 text-blue-700">Cost Savings</h3>
                <p className="text-gray-700 leading-relaxed">
                  Immunotherapy costs $150K+ per patient. Our predictions help avoid ineffective treatments, saving
                  healthcare systems billions annually.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Business Model */}
        <section className="container mx-auto px-4 mb-24 bg-gradient-to-b from-white/50 to-cyan-50/30 -mx-4 px-4 py-24">
          <div className="max-w-4xl mx-auto">
            <h2 className="text-3xl font-bold mb-12 text-center bg-gradient-to-r from-cyan-600 to-blue-600 bg-clip-text text-transparent">
              Business Model
            </h2>
            <div className="space-y-8">
              <div className="bg-white/80 border border-cyan-200 rounded-lg p-6 shadow-sm">
                <h3 className="text-xl font-semibold mb-3 text-cyan-700">B2B SaaS Platform</h3>
                <p className="text-gray-700 leading-relaxed mb-4">
                  Cloud-based prediction platform licensed to hospitals, cancer centers, and pharmaceutical companies on
                  a subscription basis.
                </p>
                <ul className="space-y-2 text-sm text-gray-700">
                  <li className="flex items-start gap-2">
                    <span className="text-cyan-600 mt-1">→</span>
                    <span>Tiered pricing based on patient volume</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-teal-600 mt-1">→</span>
                    <span>API integration with existing EHR systems</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-blue-600 mt-1">→</span>
                    <span>Continuous model updates and improvements included</span>
                  </li>
                </ul>
              </div>

              <div className="bg-white/80 border border-cyan-200 rounded-lg p-6 shadow-sm">
                <h3 className="text-xl font-semibold mb-3 text-teal-700">Clinical Trials Partnership</h3>
                <p className="text-gray-700 leading-relaxed mb-4">
                  Partner with pharmaceutical companies to optimize patient selection for immunotherapy clinical trials,
                  reducing costs and accelerating drug development timelines.
                </p>
              </div>

              <div className="bg-white/80 border border-cyan-200 rounded-lg p-6 shadow-sm">
                <h3 className="text-xl font-semibold mb-3 text-blue-700">Diagnostic Testing Service</h3>
                <p className="text-gray-700 leading-relaxed mb-4">
                  Direct-to-laboratory service processing tumor samples and delivering prediction reports to oncologists
                  within 48 hours.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Revenue Projections */}
        <section className="container mx-auto px-4 mb-24">
          <div className="max-w-4xl mx-auto">
            <h2 className="text-3xl font-bold mb-8 text-center bg-gradient-to-r from-teal-600 to-cyan-600 bg-clip-text text-transparent">
              Revenue Projections
            </h2>
            <div className="bg-gradient-to-br from-cyan-100/40 to-teal-100/40 border border-cyan-200 rounded-lg aspect-video flex items-center justify-center mb-8 shadow-sm">
              <div className="text-center p-8">
                <p className="text-sm text-cyan-700">Chart placeholder: 5-Year Revenue Projection</p>
                <p className="text-xs text-cyan-600/60 mt-2">Upload your chart/graph here</p>
              </div>
            </div>
            <div className="grid md:grid-cols-3 gap-6 text-center">
              <div className="bg-white/80 border border-cyan-200 rounded-lg p-6 shadow-sm">
                <p className="text-sm text-cyan-600 mb-2">Year 1</p>
                <p className="text-3xl font-bold bg-gradient-to-r from-cyan-600 to-blue-600 bg-clip-text text-transparent">
                  $2M
                </p>
              </div>
              <div className="bg-white/80 border border-teal-200 rounded-lg p-6 shadow-sm">
                <p className="text-sm text-teal-600 mb-2">Year 3</p>
                <p className="text-3xl font-bold bg-gradient-to-r from-teal-600 to-cyan-600 bg-clip-text text-transparent">
                  $15M
                </p>
              </div>
              <div className="bg-white/80 border border-blue-200 rounded-lg p-6 shadow-sm">
                <p className="text-sm text-blue-600 mb-2">Year 5</p>
                <p className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent">
                  $45M
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Target Customers */}
        <section className="container mx-auto px-4 mb-24">
          <div className="max-w-6xl mx-auto">
            <h2 className="text-3xl font-bold mb-12 text-center bg-gradient-to-r from-cyan-600 to-blue-600 bg-clip-text text-transparent">
              Target Customers
            </h2>
            <div className="grid md:grid-cols-2 gap-8">
              <div className="bg-white/80 border border-cyan-200 rounded-lg p-8 shadow-sm">
                <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-500 flex items-center justify-center mb-4">
                  <Building2 className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-xl font-semibold mb-3 text-cyan-700">Healthcare Providers</h3>
                <ul className="space-y-3 text-gray-700">
                  <li className="flex items-start gap-2">
                    <span className="text-cyan-600 mt-1">•</span>
                    <span>Academic medical centers and comprehensive cancer centers</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-teal-600 mt-1">•</span>
                    <span>Community oncology practices with high patient volumes</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-blue-600 mt-1">•</span>
                    <span>Hospital systems seeking to optimize oncology outcomes</span>
                  </li>
                </ul>
              </div>

              <div className="bg-white/80 border border-cyan-200 rounded-lg p-8 shadow-sm">
                <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-teal-500 to-cyan-500 flex items-center justify-center mb-4">
                  <Target className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-xl font-semibold mb-3 text-teal-700">Pharmaceutical & Biotech</h3>
                <ul className="space-y-3 text-gray-700">
                  <li className="flex items-start gap-2">
                    <span className="text-cyan-600 mt-1">•</span>
                    <span>Immunotherapy developers conducting clinical trials</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-teal-600 mt-1">•</span>
                    <span>Diagnostic companies seeking companion diagnostics</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-blue-600 mt-1">•</span>
                    <span>Contract research organizations (CROs)</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Competitive Advantage */}
        <section className="container mx-auto px-4 mb-24 bg-gradient-to-r from-cyan-500 to-blue-500 -mx-4 px-4 py-24">
          <div className="max-w-4xl mx-auto text-center">
            <h2 className="text-3xl font-bold mb-6 text-white">Our Competitive Advantage</h2>
            <p className="text-lg mb-12 text-white/90">What sets OncoMap apart in the precision oncology landscape</p>
            <div className="grid md:grid-cols-2 gap-6 text-left">
              <div className="bg-white/10 backdrop-blur rounded-lg p-6 border border-white/20">
                <h3 className="font-semibold mb-2 text-white">Proprietary Algorithms</h3>
                <p className="text-sm text-white/80">
                  Machine learning models trained on the largest curated dataset of tumor transcriptomics and treatment
                  outcomes.
                </p>
              </div>
              <div className="bg-white/10 backdrop-blur rounded-lg p-6 border border-white/20">
                <h3 className="font-semibold mb-2 text-white">Clinical Validation</h3>
                <p className="text-sm text-white/80">
                  Prospective clinical studies demonstrating improved outcomes and cost savings across multiple cancer
                  types.
                </p>
              </div>
              <div className="bg-white/10 backdrop-blur rounded-lg p-6 border border-white/20">
                <h3 className="font-semibold mb-2 text-white">Rapid Turnaround</h3>
                <p className="text-sm text-white/80">
                  Predictions delivered within 48 hours, enabling timely treatment decisions without delaying care.
                </p>
              </div>
              <div className="bg-white/10 backdrop-blur rounded-lg p-6 border border-white/20">
                <h3 className="font-semibold mb-2 text-white">Easy Integration</h3>
                <p className="text-sm text-white/80">
                  Seamless API integration with existing laboratory and EHR systems, minimizing workflow disruption.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section className="container mx-auto px-4 mb-16">
          <div className="max-w-3xl mx-auto text-center bg-white/80 border border-cyan-200 rounded-lg p-12 shadow-sm">
            <h2 className="text-3xl font-bold mb-4 bg-gradient-to-r from-teal-600 to-cyan-600 bg-clip-text text-transparent">
              Partner With Us
            </h2>
            <p className="text-lg text-gray-700 mb-8">
              Interested in bringing OncoMap to your institution or learning more about partnership opportunities?
            </p>
            <Button
              size="lg"
              className="bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-700 hover:to-blue-700 text-white"
            >
              Contact Our Team
            </Button>
          </div>
        </section>
      </div>

      {/* Footer */}
      <footer className="py-8 border-t border-cyan-200 bg-white/50">
        <div className="container mx-auto px-4 text-center text-sm text-gray-600">
          <p>&copy; 2026 OncoMap. All rights reserved.</p>
        </div>
      </footer>
    </div>
  )
}
