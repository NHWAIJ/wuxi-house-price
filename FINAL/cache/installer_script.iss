; 房价预测系统 — Inno Setup 安装包脚本(完整应用安装引导)
; 编译: 编译安装包.bat 或 ISCC.exe installer_script.iss

#define MyAppName "无锡住宅房价预测系统"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Wuxi Housing"

[Setup]
AppId={{8F3B2C1E-9A4D-4F6B-B7C2-5E1D0A3F9C46}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL=https://www.python.org/
AppSupportURL=https://www.python.org/downloads/
DefaultDirName={autopf}\WuxiHousePrice
; 显式启用选择安装位置页,且每次安装都重新询问路径(不用上次的)
DisableDirPage=no
UsePreviousAppDir=no
DefaultGroupName=房价预测系统
OutputBaseFilename=房价预测系统_安装包
OutputDir=dist_setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\web.bat
UninstallDisplayName={#MyAppName}
VersionInfoVersion=1.0.0
; 卸载向导(标准):控制面板"应用和功能"可卸载 + 开始菜单卸载项

; ---- 安装引导页 ----
; 欢迎页显示自定义信息(替换默认说明)
WizardImageFile=compiler:WizModernImage.bmp
; 许可协议页(使用 MIT LICENSE.txt)
LicenseFile=LICENSE.txt
; 信息页(安装说明)
InfoBeforeFile=INSTALL_INFO.txt

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加任务:"
Name: "autostart"; Description: "安装完成后启动网页演示"; GroupDescription: "附加任务:"; Flags: unchecked

[Files]
; 数据(372 训练小区 + 2 测试 + 属性 + 宏观)
Source: "data\*"; DestDir: "{app}\data"; Flags: recursesubdirs createallsubdirs
; 代码
Source: "scripts\*"; DestDir: "{app}\scripts"; Flags: recursesubdirs createallsubdirs
Source: "web\*"; DestDir: "{app}\web"; Flags: recursesubdirs createallsubdirs
; 集中配置
Source: "config.yaml"; DestDir: "{app}"
; 启动脚本与文档
Source: "..\*.bat"; DestDir: "{app}"
Source: "..\README.md"; DestDir: "{app}"
Source: "..\requirements.txt"; DestDir: "{app}"
Source: "..\Dockerfile"; DestDir: "{app}"
Source: "answer.md"; DestDir: "{app}"
Source: "LICENSE.txt"; DestDir: "{app}"

[Icons]
Name: "{group}\启动网页演示"; Filename: "{app}\web.bat"
Name: "{group}\训练模型(首次必跑)"; Filename: "{app}\train.bat"
Name: "{group}\卸载 房价预测系统"; Filename: "{uninstallexe}"
Name: "{autodesktop}\房价预测系统"; Filename: "{app}\web.bat"; Tasks: desktopicon

[Run]
Filename: "{app}\web.bat"; Description: "启动网页演示"; Flags: nowait postinstall skipifsilent; Tasks: autostart

[Code]
function PythonExists: Boolean;
var
  ResultCode: Integer;
begin
  Result := Exec('cmd.exe', '/c python --version >nul 2>&1', '',
                 SW_HIDE, ewWaitUntilTerminated, ResultCode);
  Result := Result and (ResultCode = 0);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    if not PythonExists then
      MsgBox('未检测到 Python 环境。' + #13#10 + #13#10 +
             '请访问 https://www.python.org/downloads/ 安装 Python 3.9+(勾选 Add to PATH),' + #13#10 +
             '然后打开命令行执行: python -m pip install -r requirements.txt 安装全部依赖库。',
             mbInformation, MB_OK);
  end;
end;
