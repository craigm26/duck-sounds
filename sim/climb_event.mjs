// The event-triggered tail lives in climb/event.mjs. This file is the name the
// shared climb scorer imports it under, for the reason stairs.js exists: the
// path has to resolve next to the core in sim/ AND next to the core in the
// phone bundle's flat /assets/, where there is no `../climb`.
export * from '../climb/event.mjs';
