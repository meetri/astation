// Agent Station dashboard tab: project and session counts, each project's
// sessions, and removing a session from a project.
//
// Plain JS, no build step: the dashboard loads this with a <script> tag and
// hands us React and its UI kit through window.__HERMES_PLUGIN_SDK__.
//
// The plugin's name is read from the tag's data-hermes-plugin attribute rather
// than written here, because the release build renames trg-researcher to
// astation and does not rewrite .js files. The name is also the API mount path.
(function () {
  "use strict";

  const SDK = window.__HERMES_PLUGIN_SDK__;
  const PLUGINS = window.__HERMES_PLUGINS__;
  if (!SDK || !PLUGINS) return;

  const script = document.currentScript;
  const NAME = (script && script.getAttribute("data-hermes-plugin")) || "trg-researcher";
  const API = "/api/plugins/" + NAME + "/api";

  const { React, fetchJSON } = SDK;
  const { useState, useEffect, useCallback } = SDK.hooks;
  const {
    Card, CardHeader, CardTitle, CardContent, Badge, Button, ConfirmDialog, Toast,
    Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter,
    Input, Label, Select, SelectOption,
  } = SDK.components;
  const h = React.createElement;

  function when(value) {
    if (value === null || value === undefined || value === "") return "—";
    // Hermes reports epoch seconds; the workspace reports ISO strings.
    if (typeof value === "number") return SDK.utils.timeAgo(value);
    return SDK.utils.isoTimeAgo(value);
  }

  // The dashboard's own "resume in chat" link. The profile rides along because
  // a session lives in one profile's store, and a ?profile= deep link wins
  // over whatever profile the dashboard currently has selected.
  function chatHref(session) {
    const base = (window.__HERMES_BASE_PATH__ || "").replace(/\/+$/, "");
    const params = new URLSearchParams({ resume: session.id, profile: session.profile || "default" });
    return (base && !base.startsWith("/") ? "/" : "") + base + "/chat?" + params.toString();
  }

  // The SDK exposes no navigate(); the host is a BrowserRouter, which follows
  // popstate, so push the URL and announce it. Modified clicks fall through to
  // the browser so "open in new tab" still works.
  function goToChat(session) {
    window.history.pushState(null, "", chatHref(session));
    window.dispatchEvent(new PopStateEvent("popstate"));
  }

  function openInChat(event, session) {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    goToChat(session);
  }

  // Profiles a new session can be created on. A profile the gateway reports as
  // disconnected is left out: creating on it is a 503. `connected: null` means
  // not checked yet, so it stays in and the route answers for it.
  let profilesPromise = null;
  function loadProfiles() {
    if (!profilesPromise) {
      profilesPromise = fetchJSON(API + "/profiles")
        .then((body) => (body.profiles || [])
          .filter((p) => p && p.name && p.connected !== false)
          .map((p) => p.name))
        .catch(() => ["default"])
        .then((names) => (names.includes("default") ? names : ["default"].concat(names)));
    }
    return profilesPromise;
  }

  // Hermes saves a session only once it has content, so the route needs a
  // first message and sends it as the session's first turn.
  function NewSessionDialog(props) {
    const { project, onClose, onCreated } = props;
    const [title, setTitle] = useState("");
    const [profile, setProfile] = useState("default");
    const [message, setMessage] = useState("");
    const [profiles, setProfiles] = useState(["default"]);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => { loadProfiles().then(setProfiles); }, []);

    const ready = title.trim() && message.trim() && !busy;

    function create() {
      if (!ready) return;
      setBusy(true);
      setError(null);
      fetchJSON(API + "/projects/" + encodeURIComponent(project.id) + "/sessions/new", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: title.trim(), first_message: message, profile: profile }),
      })
        .then((row) => onCreated(project, row))
        .catch((e) => { setError(String(e.message || e)); setBusy(false); });
    }

    return h(Dialog, { open: true, onOpenChange: (o) => { if (!o && !busy) onClose(); } },
      h(DialogContent, { className: "max-w-lg" },
        h(DialogHeader, null,
          h(DialogTitle, null, "New session"),
          h(DialogDescription, null, "Filed in " + project.title + ". Opens in chat once created.")),
        h("div", { className: "astation-form" },
          h(Label, { htmlFor: "astation-new-title" }, "Title"),
          h(Input, {
            id: "astation-new-title",
            autoFocus: true,
            value: title,
            disabled: busy,
            onChange: (e) => setTitle(e.target.value),
          }),
          h(Label, { htmlFor: "astation-new-profile" }, "Profile"),
          h(Select, { id: "astation-new-profile", value: profile, onValueChange: setProfile },
            profiles.map((name) => h(SelectOption, { key: name, value: name }, name))),
          h(Label, { htmlFor: "astation-new-message" }, "First message"),
          h("textarea", {
            id: "astation-new-message",
            className: "astation-textarea",
            rows: 5,
            value: message,
            disabled: busy,
            placeholder: "What should this session start on?",
            onChange: (e) => setMessage(e.target.value),
            onKeyDown: (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) create(); },
          }),
          error && h("div", { className: "astation-error astation-small" }, error)),
        h(DialogFooter, null,
          h(Button, { type: "button", outlined: true, disabled: busy, onClick: onClose }, "Cancel"),
          h(Button, { type: "button", disabled: !ready, onClick: create },
            busy ? "Creating…" : "Create and open"))));
  }

  function Metric(props) {
    return h(Card, null,
      h(CardContent, { className: "astation-metric" },
        h("div", { className: "astation-metric-value" }, props.value),
        h("div", { className: "astation-metric-label" }, props.label)));
  }

  function SessionList(props) {
    const { project, data, onRemove } = props;
    if (data === undefined) return h("div", { className: "astation-muted astation-pad" }, "Loading…");
    if (data.error) return h("div", { className: "astation-error astation-pad" }, data.error);
    const sessions = data.sessions || [];
    const filed = sessions.filter((s) => s.filed);
    return h("div", { className: "astation-sessions" },
      data.runtime_available === false &&
        h("div", { className: "astation-muted astation-pad" },
          "Hermes is unreachable, so titles and message counts may be missing."),
      filed.length === 0 &&
        h("div", { className: "astation-muted astation-pad" }, "No sessions in this project."),
      filed.map((s) =>
        h("div", { key: s.row_key || s.id, className: "astation-session" },
          h("div", { className: "astation-session-main" },
            s.missing === true
              ? h("div", { className: "astation-session-title" }, s.title || s.id)
              : h("a", {
                  className: "astation-session-title astation-link",
                  href: chatHref(s),
                  title: "Open in chat",
                  onClick: (e) => openInChat(e, s),
                }, s.title || s.id),
            h("div", { className: "astation-muted astation-small" },
              [
                s.profile && "profile " + s.profile,
                s.message_count != null && s.message_count + " messages",
                "started " + when(s.started_at),
                "filed " + when(s.filed_at),
              ].filter(Boolean).join(" · "))),
          s.missing === true && h(Badge, null, "gone from Hermes"),
          s.archived && h(Badge, null, "archived"),
          h(Button, {
            size: "sm",
            ghost: true,
            onClick: () => onRemove(project, s),
          }, "Remove"))),
      sessions.length > filed.length &&
        h("div", { className: "astation-muted astation-small astation-pad" },
          sessions.length - filed.length +
            " archived snapshot(s) not shown (sessions already removed or deleted)."));
  }

  function AgentStationPage() {
    const [projects, setProjects] = useState(null);
    const [error, setError] = useState(null);
    const [open, setOpen] = useState({});
    const [sessionsBy, setSessionsBy] = useState({});
    const [pending, setPending] = useState(null);
    const [removing, setRemoving] = useState(false);
    const [creatingIn, setCreatingIn] = useState(null);
    const { showToast, toast } = SDK.hooks.useToast();

    const loadProjects = useCallback(() => {
      return fetchJSON(API + "/projects")
        .then((body) => { setProjects(body.projects || []); setError(null); })
        .catch((e) => setError(String(e.message || e)));
    }, []);

    const loadSessions = useCallback((projectId) => {
      return fetchJSON(API + "/projects/" + encodeURIComponent(projectId) + "/sessions")
        .then((body) => setSessionsBy((m) => Object.assign({}, m, { [projectId]: body })))
        .catch((e) => setSessionsBy((m) =>
          Object.assign({}, m, { [projectId]: { error: String(e.message || e) } })));
    }, []);

    useEffect(() => { loadProjects(); }, [loadProjects]);

    function toggle(projectId) {
      const next = !open[projectId];
      setOpen(Object.assign({}, open, { [projectId]: next }));
      if (next && sessionsBy[projectId] === undefined) loadSessions(projectId);
    }

    function confirmRemove() {
      const { project, session } = pending;
      setRemoving(true);
      // Unfiles only: the Hermes session and its transcript stay where they are.
      fetchJSON(
        API + "/projects/" + encodeURIComponent(project.id) +
          "/sessions/" + encodeURIComponent(session.id),
        { method: "DELETE" })
        .then(() => {
          showToast("Removed \"" + (session.title || session.id) + "\" from " + project.title, "success");
          return Promise.all([loadSessions(project.id), loadProjects()]);
        })
        .catch((e) => showToast("Remove failed: " + (e.message || e), "error"))
        .finally(() => { setRemoving(false); setPending(null); });
    }

    // The row is filed and its first turn submitted; chat picks up the running turn.
    function sessionCreated(project, row) {
      setCreatingIn(null);
      showToast("Created \"" + row.title + "\" in " + project.title, "success");
      loadProjects();
      if (sessionsBy[project.id] !== undefined) loadSessions(project.id);
      goToChat({ id: row.stored_session_id || row.id, profile: row.profile });
    }

    if (error) {
      return h(Card, null, h(CardContent, { className: "astation-error astation-pad" },
        "Could not load projects: " + error));
    }
    if (projects === null) return h("div", { className: "astation-muted astation-pad" }, "Loading…");

    const totalSessions = projects.reduce((n, p) => n + (p.session_count || 0), 0);
    const empty = projects.filter((p) => !p.session_count).length;

    return h("div", { className: "astation" },
      h("div", { className: "astation-metrics" },
        h(Metric, { label: "Projects", value: projects.length }),
        h(Metric, { label: "Sessions in projects", value: totalSessions }),
        h(Metric, { label: "Empty projects", value: empty })),
      h(Card, null,
        h(CardHeader, null, h(CardTitle, null, "Projects")),
        h(CardContent, null,
          projects.length === 0 && h("div", { className: "astation-muted" }, "No projects yet."),
          projects.map((p) =>
            h("div", { key: p.id, className: "astation-project" },
              h("div", { className: "astation-project-head" },
                h("button", {
                  type: "button",
                  className: "astation-project-row",
                  "aria-expanded": !!open[p.id],
                  onClick: () => toggle(p.id),
                },
                  h("span", { className: "astation-caret" }, open[p.id] ? "▾" : "▸"),
                  h("span", { className: "astation-project-title" }, p.title),
                  (p.tags || []).map((t) => h(Badge, { key: t }, t)),
                  h("span", { className: "astation-muted astation-small" }, "updated " + when(p.updated_at)),
                  h("span", { className: "astation-count" },
                    p.session_count + (p.session_count === 1 ? " session" : " sessions"))),
                h(Button, {
                  size: "sm",
                  ghost: true,
                  title: "New session in " + p.title,
                  onClick: () => setCreatingIn(p),
                }, "+ New session")),
              open[p.id] && h(SessionList, {
                project: p,
                data: sessionsBy[p.id],
                onRemove: (project, session) => setPending({ project, session }),
              }))))),
      h(ConfirmDialog, {
        open: pending !== null,
        title: "Remove session from project?",
        description: pending
          ? "\"" + (pending.session.title || pending.session.id) + "\" will no longer be filed in " +
            pending.project.title + ". The session and its transcript stay in Hermes."
          : "",
        confirmLabel: "Remove",
        destructive: true,
        loading: removing,
        onCancel: () => { if (!removing) setPending(null); },
        onConfirm: confirmRemove,
      }),
      creatingIn && h(NewSessionDialog, {
        key: creatingIn.id,
        project: creatingIn,
        onClose: () => setCreatingIn(null),
        onCreated: sessionCreated,
      }),
      h(Toast, { toast: toast }));
  }

  PLUGINS.register(NAME, AgentStationPage);
})();
