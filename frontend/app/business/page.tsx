import Link from "next/link"
import { Button } from "@/components/ui/button"
import { ArrowLeft, TrendingUp, Users, Building2 } from "lucide-react"
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

        {/* Market & Customers */}
        <section className="container mx-auto px-4 mb-24">
          <div className="max-w-6xl mx-auto">
            <h2 className="text-3xl font-bold mb-12 text-center bg-gradient-to-r from-teal-600 to-cyan-600 bg-clip-text text-transparent">
              Market & Target Customers
            </h2>
            <div className="grid md:grid-cols-3 gap-8 mb-12">
              <div className="bg-gradient-to-br from-cyan-100/40 to-blue-100/30 border border-cyan-200 rounded-lg p-6 shadow-sm">
                <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-500 flex items-center justify-center mb-4">
                  <TrendingUp className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-xl font-semibold mb-3 text-cyan-700">Target Market</h3>
                <p className="text-gray-700 leading-relaxed">
                  Head-and-neck cancer patients, affecting <strong>1.4 million patients globally</strong> each year and
                  comprising 7.6% of all cancers with the lowest immunotherapy response rates.
                </p>
              </div>

              <div className="bg-gradient-to-br from-teal-100/40 to-cyan-100/30 border border-cyan-200 rounded-lg p-6 shadow-sm">
                <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-teal-500 to-cyan-500 flex items-center justify-center mb-4">
                  <Building2 className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-xl font-semibold mb-3 text-teal-700">Primary Buyers</h3>
                <p className="text-gray-700 leading-relaxed">
                  Clinical institutions: academic medical centers, comprehensive cancer centers, and regional hospitals
                  that administer immunotherapy.
                </p>
              </div>

              <div className="bg-gradient-to-br from-blue-100/30 to-teal-100/40 border border-cyan-200 rounded-lg p-6 shadow-sm">
                <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-blue-500 to-teal-500 flex items-center justify-center mb-4">
                  <Users className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-xl font-semibold mb-3 text-blue-700">End Users</h3>
                <p className="text-gray-700 leading-relaxed">
                  Oncologists and pathologists who make treatment decisions, seeking to validate tumor board decisions
                  and reduce treatment uncertainty.
                </p>
              </div>
            </div>

            <div className="bg-white/80 border border-cyan-200 rounded-lg p-8 shadow-sm">
              <h3 className="text-2xl font-semibold mb-4 text-cyan-700">Customer Value Proposition</h3>
              <div className="space-y-4 text-gray-700">
                <p className="leading-relaxed">
                  OncoMap's clinician customers value resources that reduce uncertainty around immunotherapy response,
                  validate treatment selection, and decrease the likelihood of incorrect decisions to reduce unwarranted
                  treatment costs.
                </p>
                <p className="leading-relaxed">
                  Institutional buyers prioritize{" "}
                  <strong>cost containment, efficacy improvement, and operational efficiency</strong>. With PD-1
                  immunotherapy costing over <strong>$150,000 per patient</strong>, modest accuracy improvements
                  translate into significant savings.
                </p>
                <p className="leading-relaxed">
                  OncoMap fits into the broader ecosystem as a software-based clinical decision support layer that
                  complements existing diagnostics rather than replacing them, maximizing accuracy and precision.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Customer Acquisition */}
        <section className="container mx-auto px-4 mb-24">
          <div className="max-w-6xl mx-auto">
            <h2 className="text-3xl font-bold mb-12 text-center bg-gradient-to-r from-teal-600 to-cyan-600 bg-clip-text text-transparent">
              Customer Acquisition Strategy
            </h2>

            <div className="space-y-8">
              <div className="bg-white/80 border border-cyan-200 rounded-lg p-8 shadow-sm">
                <div className="flex items-start gap-4 mb-4">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-br from-cyan-500 to-blue-500 flex items-center justify-center text-white font-bold flex-shrink-0">
                    1
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold mb-3 text-cyan-700">Initial Pilot & Validation</h3>
                    <p className="text-gray-700 leading-relaxed mb-3">
                      <strong>Fred Hutchinson Cancer Center</strong> (Bellevue, Washington) serves as our initial
                      traction point. OncoMap's team has direct support from Dr. Holland and interns at this leading
                      cancer center.
                    </p>
                    <p className="text-gray-700 leading-relaxed">
                      This enables rapid internal validation, clinician feedback, and development of trust and
                      credibility within the industry.
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-white/80 border border-cyan-200 rounded-lg p-8 shadow-sm">
                <div className="flex items-start gap-4 mb-4">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-br from-teal-500 to-cyan-500 flex items-center justify-center text-white font-bold flex-shrink-0">
                    2
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold mb-3 text-teal-700">Early Adoption Expansion</h3>
                    <p className="text-gray-700 leading-relaxed mb-3">
                      <strong>UW Medicine</strong> expansion leveraging close clinical and research ties with Fred Hutch
                      to introduce OncoMap as a low-risk pilot tool for head-and-neck cancer care.
                    </p>
                    <p className="text-gray-700 leading-relaxed">
                      Attraction reinforced through validation studies, clinician-facing demonstrations, and case
                      studies showing improved immunotherapy decision-making.
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-white/80 border border-cyan-200 rounded-lg p-8 shadow-sm">
                <div className="flex items-start gap-4 mb-4">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-teal-500 flex items-center justify-center text-white font-bold flex-shrink-0">
                    3
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold mb-3 text-blue-700">Scale Through Partnerships</h3>
                    <p className="text-gray-700 leading-relaxed">
                      <strong>Direct institutional licensing</strong> targeting cancer center departments rather than
                      individual oncologists. Strategic partnerships with academic medical centers, sequencing
                      providers, and oncology networks to scale adoption efficiently.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Revenue Model */}
        <section className="container mx-auto px-4 mb-24 bg-gradient-to-b from-white/50 to-teal-50/30 -mx-4 px-4 py-24">
          <div className="max-w-6xl mx-auto">
            <h2 className="text-3xl font-bold mb-12 text-center bg-gradient-to-r from-cyan-600 to-blue-600 bg-clip-text text-transparent">
              Revenue Model
            </h2>

            <div className="bg-white/80 border border-cyan-200 rounded-lg p-8 shadow-sm mb-8">
              <h3 className="text-2xl font-semibold mb-4 text-cyan-700">Tiered Annual Subscription Model</h3>
              <p className="text-gray-700 leading-relaxed mb-4">
                OncoMap generates revenue through institutional subscriptions with token-based patient analysis credits,
                creating predictable recurring revenue that scales with patient volume.
              </p>
            </div>

            <div className="grid md:grid-cols-2 gap-8 mb-12">
              <div className="bg-gradient-to-br from-cyan-100/40 to-blue-100/30 border-2 border-cyan-300 rounded-lg p-8 shadow-sm">
                <div className="mb-4">
                  <h3 className="text-2xl font-bold text-cyan-700 mb-2">Tier 1: Immunotherapy Response Prediction</h3>
                  <p className="text-sm text-cyan-600">Higher-value product for PD-1 treatment decisions</p>
                </div>
                <div className="space-y-4">
                  <div>
                    <p className="text-3xl font-bold text-cyan-700 mb-1">$1,000/year</p>
                    <p className="text-sm text-gray-600">Base subscription includes 50 tokens</p>
                  </div>
                  <div className="border-t border-cyan-200 pt-4">
                    <p className="text-2xl font-semibold text-cyan-600 mb-1">$5 per token</p>
                    <p className="text-sm text-gray-600">Additional patient analyses</p>
                  </div>
                  <div className="bg-white/60 rounded p-4 mt-4">
                    <p className="text-sm text-gray-700 leading-relaxed">
                      <strong>Value:</strong> Directly informs treatment decisions, helping save $150,000 per patient on
                      ineffective immunotherapy
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-br from-teal-100/40 to-cyan-100/30 border-2 border-teal-300 rounded-lg p-8 shadow-sm">
                <div className="mb-4">
                  <h3 className="text-2xl font-bold text-teal-700 mb-2">Tier 2: Patient Analysis (KNN)</h3>
                  <p className="text-sm text-teal-600">Routine treatment planning and research</p>
                </div>
                <div className="space-y-4">
                  <div>
                    <p className="text-3xl font-bold text-teal-700 mb-1">$500/year</p>
                    <p className="text-sm text-gray-600">Base subscription includes 50 tokens</p>
                  </div>
                  <div className="border-t border-teal-200 pt-4">
                    <p className="text-2xl font-semibold text-teal-600 mb-1">$3 per token</p>
                    <p className="text-sm text-gray-600">Additional patient analyses</p>
                  </div>
                  <div className="bg-white/60 rounded p-4 mt-4">
                    <p className="text-sm text-gray-700 leading-relaxed">
                      <strong>Value:</strong> Supports broader use in routine treatment planning and research
                      applications
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-white/80 border border-cyan-200 rounded-lg p-8 shadow-sm">
              <h3 className="text-2xl font-semibold mb-6 text-cyan-700">Financial Overview</h3>
              <div className="grid md:grid-cols-3 gap-6 mb-8">
                <div className="text-center p-4 bg-gradient-to-br from-cyan-100/40 to-blue-100/30 rounded-lg">
                  <p className="text-sm text-cyan-600 mb-2">Initial Startup Costs</p>
                  <p className="text-3xl font-bold text-cyan-700">$10,000</p>
                  <p className="text-xs text-gray-600 mt-2">Cloud, deployment, datasets</p>
                </div>
                <div className="text-center p-4 bg-gradient-to-br from-teal-100/40 to-cyan-100/30 rounded-lg">
                  <p className="text-sm text-teal-600 mb-2">Annual Operating Costs</p>
                  <p className="text-3xl font-bold text-teal-700">$2-3K</p>
                  <p className="text-xs text-gray-600 mt-2">Hosting, compute, maintenance</p>
                </div>
                <div className="text-center p-4 bg-gradient-to-br from-blue-100/30 to-teal-100/40 rounded-lg">
                  <p className="text-sm text-blue-600 mb-2">Per-Patient Cost</p>
                  <p className="text-3xl font-bold text-blue-700">$0.05-0.10</p>
                  <p className="text-xs text-gray-600 mt-2">Minimal delivery costs</p>
                </div>
              </div>

              <div className="bg-gradient-to-r from-cyan-500 to-blue-500 rounded-lg p-6 text-white">
                <h4 className="text-xl font-semibold mb-3">Path to Profitability</h4>
                <p className="mb-2">
                  <strong>Combined annual revenue per institution:</strong> ~$1,000-$1,200
                </p>
                <p className="mb-2">
                  <strong>Break-even point:</strong> 5-10 institutions
                </p>
                <p>
                  <strong>Timeline:</strong> Achievable within first year
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Funding Requirements */}
        <section className="container mx-auto px-4 mb-24">
          <div className="max-w-6xl mx-auto">
            <h2 className="text-3xl font-bold mb-12 text-center bg-gradient-to-r from-teal-600 to-cyan-600 bg-clip-text text-transparent">
              Funding Requirements
            </h2>

            <div className="bg-white/80 border border-cyan-200 rounded-lg p-8 shadow-sm mb-8">
              <div className="text-center mb-8">
                <p className="text-5xl font-bold bg-gradient-to-r from-cyan-600 to-blue-600 bg-clip-text text-transparent mb-2">
                  $10-15K
                </p>
                <p className="text-lg text-gray-700">Initial funding for one-year development and pilot deployment</p>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-6 mb-8">
              <div className="bg-gradient-to-br from-cyan-100/40 to-blue-100/30 border border-cyan-200 rounded-lg p-6 shadow-sm">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-16 h-16 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-500 flex items-center justify-center">
                    <span className="text-2xl font-bold text-white">30%</span>
                  </div>
                  <h3 className="text-xl font-semibold text-cyan-700">Compute & Infrastructure</h3>
                </div>
                <p className="text-gray-700 leading-relaxed">
                  Cloud-based CPU compute for RNA-seq processing, model training/testing, and secure data storage. Costs
                  may decrease with efficient penalized regressions and k-nearest-neighbors models.
                </p>
              </div>

              <div className="bg-gradient-to-br from-teal-100/40 to-cyan-100/30 border border-teal-200 rounded-lg p-6 shadow-sm">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-16 h-16 rounded-lg bg-gradient-to-br from-teal-500 to-cyan-500 flex items-center justify-center">
                    <span className="text-2xl font-bold text-white">20%</span>
                  </div>
                  <h3 className="text-xl font-semibold text-teal-700">Data Acquisition</h3>
                </div>
                <p className="text-gray-700 leading-relaxed">
                  Public bulk-RNA-seq datasets with limited licensing costs. Minimized through academic access with Fred
                  Hutch Cancer Center and other partners.
                </p>
              </div>

              <div className="bg-gradient-to-br from-blue-100/30 to-teal-100/40 border border-blue-200 rounded-lg p-6 shadow-sm">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-16 h-16 rounded-lg bg-gradient-to-br from-blue-500 to-teal-500 flex items-center justify-center">
                    <span className="text-2xl font-bold text-white">25%</span>
                  </div>
                  <h3 className="text-xl font-semibold text-blue-700">Deployment</h3>
                </div>
                <p className="text-gray-700 leading-relaxed">
                  Interface refinement, backend APIs, and hosting costs to create a production-ready clinical platform.
                </p>
              </div>

              <div className="bg-gradient-to-br from-cyan-100/30 to-teal-100/40 border border-cyan-200 rounded-lg p-6 shadow-sm">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-16 h-16 rounded-lg bg-gradient-to-br from-cyan-600 to-teal-600 flex items-center justify-center">
                    <span className="text-2xl font-bold text-white">10%</span>
                  </div>
                  <h3 className="text-xl font-semibold text-cyan-700">Outreach</h3>
                </div>
                <p className="text-gray-700 leading-relaxed">
                  Clinical demonstrations, validation studies, and relationship building with institutional partners.
                </p>
              </div>
            </div>

            <div className="bg-gradient-to-br from-teal-100/40 to-blue-100/30 border border-teal-200 rounded-lg p-6 shadow-sm">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-16 h-16 rounded-lg bg-gradient-to-br from-teal-600 to-cyan-600 flex items-center justify-center">
                  <span className="text-2xl font-bold text-white">15%</span>
                </div>
                <h3 className="text-xl font-semibold text-teal-700">Licensing & IP</h3>
              </div>
              <p className="text-gray-700 leading-relaxed">
                Intellectual property consultations and licensing requirements to protect OncoMap's proprietary
                algorithms and ensure compliance.
              </p>
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

      
    </div>
  )
}
