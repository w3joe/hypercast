/** Use the supplied artwork unchanged; the viewBox removes its transparent margins. */
export default function BrandLogo({ className = '' }: { className?: string }) {
  return <svg className={`brand-logo ${className}`} viewBox="48 115 2076 436" role="img" aria-label="Hypercast" focusable="false">
    <image href="/hypercast-logo.png" width="2172" height="724" />
  </svg>
}
