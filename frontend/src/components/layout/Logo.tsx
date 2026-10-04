/** Balloon mark for the AI Studio brand — a pink-to-red balloon with a
 * curled string. */
export function Logo({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none" aria-hidden>
      <defs>
        <radialGradient id="balloon" cx="35%" cy="30%" r="75%">
          <stop offset="0" stopColor="#ff8aa8" />
          <stop offset="1" stopColor="#e11d48" />
        </radialGradient>
      </defs>
      <ellipse cx="17" cy="12" rx="9" ry="10" fill="url(#balloon)" />
      <path d="M15.5 21.5 17 24l-1.6 1.2" fill="#e11d48" stroke="#e11d48" strokeWidth="1" strokeLinejoin="round" />
      <path
        d="M17 24c-3 2-1 4-4 6s-1 2-3 1"
        stroke="#8b5cf6"
        strokeWidth="1.4"
        strokeLinecap="round"
      />
    </svg>
  )
}
