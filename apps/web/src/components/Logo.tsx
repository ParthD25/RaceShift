export function Logo() {
  return (
    <div className="brand" aria-label="RaceShift home">
      <svg className="brand-mark" viewBox="0 0 48 48" aria-hidden="true">
        <path d="M8 9h12l-4 10H4L8 9Zm16 0h12l-4 10H20l4-10ZM4 23h12l-4 10H0l4-10Zm16 0h12l-4 10H16l4-10Zm16 0h12l-4 10H32l4-10Z" fill="currentColor" />
      </svg>
      <div>
        <div className="brand-name">RaceShift</div>
        <div className="brand-sub">Data driven. Faster.</div>
      </div>
    </div>
  );
}
