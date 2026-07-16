export function Logo({ size = 36 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 48 48"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <rect width="48" height="48" rx="12" fill="#151B2F" />
      <path
        d="M14 14 C 30 14 10 24 26 24 C 42 24 22 34 38 34"
        stroke="#4338CA"
        strokeWidth="4.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M18 10 C 34 10 14 20 30 20 C 46 20 26 30 42 30"
        stroke="#10B981"
        strokeWidth="4.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}
