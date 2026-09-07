import type { ReactNode } from 'react';

export function Panel({ title, icon, action, children, className = '' }: { title: string; icon?: ReactNode; action?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`panel ${className}`}>
      <header className="panel-header">
        <div className="panel-title">{icon}{title}</div>
        {action ? <div>{action}</div> : null}
      </header>
      <div className="panel-body">{children}</div>
    </section>
  );
}
