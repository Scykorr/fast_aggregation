; Установщик fastagg. Собирается на windows-latest: ISCC.exe packaging\fastagg.iss
; Рядом должны лежать dist\fastagg.exe и, по желанию, packaging\runtimes\*.zip

#define MyAppName "fastagg"
#define MyAppVersion "0.1.0"
#define MyAppExe "fastagg.exe"

[Setup]
AppId={{7F3A9C2E-6B14-4F0A-9D55-FA57A66E1101}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=fastagg
DefaultDirName={autopf}\fastagg
DefaultGroupName=fastagg
OutputDir=..\dist\installer
#ifndef SetupBaseName
#define SetupBaseName "fastagg-setup-0.1.0"
#endif
OutputBaseFilename={#SetupBaseName}
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
SetupIconFile=fastagg.ico
UninstallDisplayIcon={app}\{#MyAppExe}
PrivilegesRequired=admin
WizardStyle=modern

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[Tasks]
Name: "rolecoord"; Description: "Автозапуск в роли координатора"; GroupDescription: "Роль этого компьютера:"; Flags: exclusive
Name: "rolenode"; Description: "Автозапуск в роли узла"; GroupDescription: "Роль этого компьютера:"; Flags: exclusive unchecked
Name: "desktopicon"; Description: "Ярлык на рабочем столе"; GroupDescription: "Дополнительно:"; Flags: checkedonce

[Files]
Source: "..\dist\fastagg.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "runtimes\*"; DestDir: "{%USERPROFILE}\.fast_aggregation\runtimes\b12359"; Flags: ignoreversion recursesubdirs skipifsourcedoesntexist uninsneveruninstall
#ifdef BundleCuda
Source: "cuda\*.zip"; DestDir: "{%USERPROFILE}\.fast_aggregation\runtimes\b12359"; Flags: ignoreversion
#endif
; Архивы CUDA, положенные рядом с setup.exe в папку cuda\, ставятся без пересборки установщика.
Source: "{src}\cuda\*"; DestDir: "{%USERPROFILE}\.fast_aggregation\runtimes\b12359"; Flags: external skipifsourcedoesntexist recursesubdirs uninsneveruninstall

[Icons]
Name: "{group}\Координатор"; Filename: "{app}\{#MyAppExe}"; Parameters: "coordinator"
Name: "{group}\Узел"; Filename: "{app}\{#MyAppExe}"; Parameters: "node"
Name: "{group}\Трей"; Filename: "{app}\{#MyAppExe}"; Parameters: "tray --role coordinator"
Name: "{autodesktop}\fastagg"; Filename: "{app}\{#MyAppExe}"; Parameters: "tray --role coordinator"; Tasks: desktopicon and rolecoord
Name: "{autodesktop}\fastagg"; Filename: "{app}\{#MyAppExe}"; Parameters: "tray --role node"; Tasks: desktopicon and rolenode

[Registry]
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "fastagg"; ValueData: """{app}\{#MyAppExe}"" tray --role coordinator"; Tasks: rolecoord; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "fastagg"; ValueData: """{app}\{#MyAppExe}"" tray --role node"; Tasks: rolenode; Flags: uninsdeletevalue

[Run]
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall add rule name=""fastagg-discovery"" dir=in action=allow protocol=UDP localport=45221 profile=private"; Flags: runhidden
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall add rule name=""fastagg-node"" dir=in action=allow protocol=TCP localport=47000 profile=private"; Flags: runhidden
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall add rule name=""fastagg-rpc"" dir=in action=allow protocol=TCP localport=50052 profile=private"; Flags: runhidden
Filename: "{cmd}"; Parameters: "/C netsh advfirewall firewall delete rule name=fastagg-dashboard & netsh advfirewall firewall add rule name=fastagg-dashboard dir=in action=allow protocol=TCP localport=47080-47099 profile=private"; Flags: runhidden
Filename: "{app}\{#MyAppExe}"; Description: "Открыть координатор"; Parameters: "coordinator"; Flags: nowait postinstall skipifsilent; Tasks: rolecoord

[UninstallRun]
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall delete rule name=""fastagg-discovery"""; Flags: runhidden
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall delete rule name=""fastagg-node"""; Flags: runhidden
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall delete rule name=""fastagg-rpc"""; Flags: runhidden
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall delete rule name=""fastagg-dashboard"""; Flags: runhidden
