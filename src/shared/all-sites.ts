import { browser } from 'wxt/browser';
import { OPTIONAL_HOST_PERMISSIONS } from './manifest';

/** Host access to every http(s) site, which the floating button on every page needs. */
export async function hasAllSitesAccess(): Promise<boolean> {
  return browser.permissions
    .contains({ origins: [...OPTIONAL_HOST_PERMISSIONS] })
    .catch(() => false);
}

/** Must run inside a user gesture on an extension page. */
export async function requestAllSitesAccess(): Promise<boolean> {
  return browser.permissions
    .request({ origins: [...OPTIONAL_HOST_PERMISSIONS] })
    .catch(() => false);
}
