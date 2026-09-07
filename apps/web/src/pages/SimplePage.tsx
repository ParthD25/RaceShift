import { Panel } from '../components/Panel';

type Props = { title: string; description: string; status?: string; detail?: string };

export default function SimplePage({ title, description, status, detail }: Props) {
  return (
    <div className="page-stack">
      <div className="page-heading">
        <div>
          <h1>
            {title} {status ? <span className="badge badge-fixture"><i />{status}</span> : null}
          </h1>
          <p>{description}</p>
        </div>
      </div>
      <Panel title="Workspace">
        <div className="empty-workspace">
          <div className="empty-ring" />
          <h2>{title} workspace</h2>
          <p>{detail ?? 'This surface is wired into the RaceShift shell and reserved for the next validated workflow, not placeholder product claims.'}</p>
        </div>
      </Panel>
    </div>
  );
}
