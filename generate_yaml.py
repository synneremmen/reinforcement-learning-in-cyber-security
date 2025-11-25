import yaml

discovery_techniques = {
    'SystemOwnerUserDiscovery',
    'ContainerandResourceDiscovery',
    'GroupPolicyDiscovery',
    'SystemChecks',
    'DomainGroups',
    'SystemServiceDiscovery',
    'NetworkSniffing',
    'NetworkShareDiscovery',
    'PeripheralDeviceDiscovery',
    'SystemInformationDiscovery',
    'ApplicationWindowDiscovery',
    'CloudInfrastructureDiscovery',
    'BrowserInformationDiscovery',
    'SystemNetworkConfigurationDiscovery',
    'DomainTrustDiscovery',
    'FileandDirectoryDiscovery',
    'SystemNetworkConnectionsDiscovery',
    'CloudStorageObjectDiscovery',
    'ProcessDiscovery',
    'LocalGroups',
    'PasswordPolicyDiscovery',
    'SystemLanguageDiscovery',
    'QueryRegistry',
    'SecuritySoftwareDiscovery',
    'CloudServiceDiscovery',
    'RemoteSystemDiscovery',
    'NetworkServiceDiscovery',
    'SoftwareDiscovery',
    'SystemTimeDiscovery'
}

all_techniques = [
    'LinuxLateralMovement', 'ExtraWindowMemoryInjection', 'ScheduledTask', 'ArchiveviaUtility', 'VNC', 'WindowsManagementInstrumentation', 'ScreenCapture', 'SystemOwnerUserDiscovery', 'Rundll32', 'ContainerandResourceDiscovery', 'StandardEncoding', 'PluggableAuthenticationModules', 'Keylogging', 'LinuxandMacFileandDirectoryPermissionsModification', 'PasswordGuessing', 'PubPrn', 'OSCredentialDumping', 'DirectVolumeAccess', 'Rootkit', 'PowerShellProfile', 'JavaScript', 'AudioCapture', 'ExternalRemoteServices', 'StealWebSessionCookie', 'ContainerOrchestrationJob', 'BypassUserAccountControl', 'SudoandSudoCaching', 'SecurityAccountManager', 'ServicesRegistryPermissionsWeakness', 'DNS', 'CloudInstanceMetadataAPI', 'GroupPolicyDiscovery', 'LocalDataStaging', 'MatchLegitimateNameorLocation', 'PasswordCracking', 'LocalEmailCollection', 'Keychain', 'BootorLogonAutostartExecution', 'LSASecrets', 'SAMLTokens', 'ServiceStop', 'DomainAccount', 'ActiveSetup', 'HideArtifacts', 'DynamicDataExchange', 'MaliciousFile', 'Hardware', 'DomainTrustModification', 'LocalAccount', 'SafeModeBoot', 'WindowsService', 'SystemChecks', 'Cron', 'DomainGroups', 'ClearLinuxorMacSystemLogs', 'OfficeApplicationStartup', 'InstallUtil', 'AdditionalCloudRoles', 'PrintProcessors', 'SpearphishingAttachment', 'DLLSearchOrderHijacking', 'AutomatedCollection', 'ClipboardData', 'ProcFilesystem', 'GatekeeperBypass', 'SystemServiceDiscovery', 'DatafromCloudStorage', 'CredentialsinRegistry', 'NetworkShareDiscovery', 'PeripheralDeviceDiscovery', 'WindowsFileandDirectoryPermissionsModification', 'Addins', 'TransportAgent', 'SystemInformationDiscovery', 'Msiexec', 'PasswordFilterDLL', 'TerminalServicesDLL', 'AppleScript', 'BrowserExtensions', 'NativeAPI', 'ASREPRoasting', 'ClearCommandHistory', 'IndirectCommandExecution', 'ReplicationThroughRemovableMedia', 'DatafromLocalSystem', 'DeobfuscateDecodeFilesorInformation', 'ImpairDefenses', 'SupplyChainCompromise', 'CredentialsfromPasswordStores', 'RemoteAccessSoftware', 'ArchiveviaLibrary', 'ThreadExecutionHijacking', 'Masquerading', 'ApplicationShimming', 'UnsecuredCredentials', 'PortMonitors', 'ClearMailboxData', 'LoginHook', 'ProcessInjection', 'SystemBinaryProxyExecution', 'Timestomp', 'ReflectiveCodeLoading', 'EscapetoHost', 'ShortcutModification', 'ApplicationWindowDiscovery', 'CMSTP', 'DisableWindowsEventLogging', 'SMBWindowsAdminShares', 'ProtocolTunneling', 'ControlPanel', 'SecuritySupportProvider', 'DisableorModifySystemFirewall', 'ArchiveCollectedData', 'SIPandTrustProviderHijacking', 'RogueDomainController', 'DeployContainer', 'ModifyRegistry', 'LaunchDaemon', 'CloudInfrastructureDiscovery', 'CredentialsfromWebBrowsers', 'PathInterceptionbySearchOrderHijacking', 'BinaryPadding', 'WebShell', 'GroupPolicyModification', 'BrowserInformationDiscovery', 'PrivateKeys', 'WindowsRemoteManagement', 'DefaultAccounts', 'TimeProviders', 'Trap', 'DynamicLinkerHijacking', 'LocalAccount', 'ClearWindowsEventLogs', 'LLMNRNBTNSPoisoningandSMBRelay', 'LSASSMemory', 'CreateProcesswithToken', 'SetuidandSetgid', 'WinlogonHelperDLL', 'DistributedComponentObjectModel', 'PasswordSpraying', 'CachedDomainCredentials', 'SSHAuthorizedKeys', 'ImageFileExecutionOptionsInjection', 'Odbcconf', 'VideoCapture', 'SystemNetworkConfigurationDiscovery', 'AccessibilityFeatures', 'IndicatorBlocking', 'DomainAccount', 'DomainTrustDiscovery', 'GoldenTicket', 'AutomatedExfiltration', 'IndicatorRemoval', 'PasstheTicket', 'ContainerAdministrationCommand', 'FileandDirectoryDiscovery', 'MasqueradeTaskorService', 'AsynchronousProcedureCall', 'PlistFileModification', 'AppCertDLLs', 'EmailForwardingRule', 'StealorForgeAuthenticationCertificates', 'SystemNetworkConnectionsDiscovery', 'MarkoftheWebBypass', 'BuildImageonHost', 'PortableExecutableInjection', 'Launchctl', 'BashHistory', 'CredentialsInFiles', 'Mshta', 'LoginItems', 'CloudStorageObjectDiscovery', 'TokenImpersonationTheft', 'StealApplicationAccessToken', 'AdditionalCloudCredentials', 'InternalDefacement', 'HiddenUsers', 'GroupPolicyPreferences', 'ExfiltrationOverAsymmetricEncryptedNonC2Protocol', 'ProcessDiscovery', 'ImpairCommandHistoryLogging', 'WindowsManagementInstrumentationEventSubscription', 'SoftwareDeploymentTools', 'ExfiltrationOverC2Channel', 'ParentPIDSpoofing', 'PowerShell', 'ChangeDefaultFileAssociation', 'Emond', 'RegistryRunKeysStartupFolder', 'CloudAccount', 'LocalGroups', 'AccountManipulation', 'ExfiltrationOverAlternativeProtocol', 'KernelModulesandExtensions', 'GUIInputCapture', 'SystemdTimers', 'CompiledHTMLFile', 'NetworkShareConnectionRemoval', 'MultihopProxy', 'UnixShell', 'DisableorModifyTools', 'InterProcessCommunication', 'DatafromNetworkSharedDrive', 'MaliciousImage', 'NonStandardPort', 'ProcessHollowing', 'AccountAccessRemoval', 'CredentialStuffing', 'ObfuscatedFilesorInformation', 'IISComponents', 'RunVirtualInstance', 'PasswordPolicyDiscovery', 'EventTriggeredExecution', 'UnixShellConfigurationModification', 'ForcedAuthentication', 'SIDHistoryInjection', 'DataEncryptedforImpact', 'EncryptedChannel', 'AuthenticationPackage', 'Regsvr32', 'ExfiltrationtoTextStorageSites', 'ComponentObjectModelHijacking', 'RenameSystemUtilities', 'OutlookHomePage', 'ExfiltrationtoCloudStorage', 'LateralToolTransfer', 'PathInterceptionbyUnquotedPath', 'StartupItems', 'SystemLanguageDiscovery', 'NonApplicationLayerProtocol', 'QueryRegistry', 'DataTransferSizeLimits', 'RegsvcsRegasm', 'InstallRootCertificate', 'CompileAfterDelivery', 'BITSJobs', 'MSBuild', 'DisableCloudLogs', 'SecuritySoftwareDiscovery', 'HiddenWindow', 'Python', 'AppInitDLLs', 'ResourceHijacking', 'Screensaver', 'etcpasswdandetcshadow', 'LaunchAgent', 'WindowsCommandShell', 'SilverTicket', 'WindowsCredentialManager', 'DataDestruction', 'HTMLSmuggling', 'FileDeletion', 'TemplateInjection', 'RCScripts', 'SoftwarePacking', 'WebProtocols', 'VisualBasic', 'SystemdService', 'RDPHijacking', 'CloudServiceDiscovery', 'RemoteSystemDiscovery', 'NetworkServiceDiscovery', 'SoftwareDiscovery', 'SpaceafterFilename', 'ReopenedApplications', 'PasstheHash', 'DLLSideLoading', 'IngressToolTransfer', 'AdditionalEmailDelegatePermissions', 'RemoteDesktopProtocol', 'LogonScriptWindows', 'XSLScriptProcessing', 'HiddenFilesandDirectories', 'OfficeTest', 'NTDS', 'LSASSDriver', 'ServiceExecution', 'CloudAccounts', 'NTFSFileAttributes', 'Kerberoasting', 'DCSync', 'SystemTimeDiscovery', 'At', 'DynamicLinkerHijacking', 'CredentialAPIHooking', 'InhibitSystemRecovery', 'NetshHelperDLL', 'InternalProxy', 'SystemScriptProxyExecution', 'ContainerAPI', 'ExfiltrationOverUnencryptedNonC2Protocol', 'LocalAccounts', 'TrustedDeveloperUtilitiesProxyExecution', 'SystemShutdownReboot', 'COR_PROFILER'
]

actions = {}
for class_name in all_techniques:

    reward = {'immediate': 20, 'recurring': 0} if class_name in discovery_techniques else {'immediate': 0, 'recurring': 0}
    actions[class_name] = {'class': class_name, 'reward': reward}

yaml_content = {'actions': actions, 'action_space': 'RedDiscreteActionSpace', 'class': 'RLARTAgent', 'rl': True}

print(yaml.dump(yaml_content, default_flow_style=False))